<#
.SYNOPSIS
    Run the test suite and every script in two checkouts of Moirai and report
    any difference in what they print.

.DESCRIPTION
    Used before every merge: Base is main, Trial is the branch being tested.
    Each side runs with its own .venv, so each imports its own code (checked
    and printed).

    The two sides are interleaved: each script runs on base and then at once
    on trial, so both see the same network and the same live FRED data. A
    network outage or a FRED revision between the runs cannot then pass for
    a code change.

    stdout and stderr are captured separately through .NET, not through
    PowerShell's 2>&1, so lines are never wrapped at the console width or
    turned into error records, and the streams cannot interleave differently.

    If the two outputs differ and either side shows a network error, both
    sides are rerun once after a pause. If they still differ, the result is
    reported as INCONCLUSIVE (network), never as a real difference.

    Before comparing, run-to-run noise is masked:
      - the two repository roots      -> <repo>
      - clock times (not 00:00:00)    -> <time>     (log lines, known_at stamps)
      - pytest durations              -> in <dur>
      - hex content hashes (16+ hex)  -> <hash>
      - memory addresses (0x...)      -> <addr>
      - trailing whitespace, CRLF/LF
    Midnight timestamps are left alone because they are data, not noise.

    Exit code: real differences plus inconclusive results, so 0 means clean.

.EXAMPLE
    powershell -NoProfile -ExecutionPolicy Bypass -File scripts\compare_runs.ps1
#>
[CmdletBinding()]
param(
    [string]   $Base  = 'C:\dev\moirai',
    [string]   $Trial = 'C:\dev\moirai-trial',
    [string]   $Out   = 'C:\dev\compare-runs',
    [string[]] $Skip  = @('fetch_2022.py'),
    [int]      $RetryPauseSeconds = 15,
    [switch]   $NoPytest
)

$ErrorActionPreference = 'Stop'
$Utf8 = New-Object System.Text.UTF8Encoding($false)
$Marker = '.compare_runs'
$NetworkPattern = 'getaddrinfo failed|Errno 11001|Name or service not known|' +
                  'Temporary failure in name resolution|ConnectError|ConnectTimeout|' +
                  'ReadTimeout|timed out|Too Many Requests|\b429\b|HTTP 5\d\d|Bad Gateway|' +
                  'Service Unavailable|Gateway Time-?out|' +
                  'not available \(IngestionError\)'

# ---------------------------------------------------------------- preflight
foreach ($h in 'api.stlouisfed.org') {
    try { [System.Net.Dns]::GetHostAddresses($h) | Out-Null }
    catch { throw "Network check failed: cannot resolve $h. Fix the connection and rerun." }
}

# ---------------------------------------------------------------- output folder
# Only ever delete a folder this tool created (it carries the marker file).
if (Test-Path $Out) {
    if (-not (Test-Path (Join-Path $Out $Marker))) {
        throw "$Out exists and was not created by this tool. Choose another -Out."
    }
    Remove-Item $Out -Recurse -Force
}
foreach ($d in 'base', 'trial', 'norm\base', 'norm\trial', 'diff') {
    New-Item -ItemType Directory -Path (Join-Path $Out $d) -Force | Out-Null
}
[System.IO.File]::WriteAllText((Join-Path $Out $Marker), '', $Utf8)

# ---------------------------------------------------------------- run a process
function Invoke-Captured {
    param([string]$Exe, [string[]]$ArgList, [string]$Cwd)

    $psi = New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName = $Exe
    $psi.Arguments = ($ArgList | ForEach-Object {
        if ($_ -match '\s') { '"' + $_ + '"' } else { $_ }
    }) -join ' '
    $psi.WorkingDirectory = $Cwd
    $psi.UseShellExecute = $false
    $psi.RedirectStandardOutput = $true
    $psi.RedirectStandardError = $true
    $psi.StandardOutputEncoding = $Utf8
    $psi.StandardErrorEncoding = $Utf8
    $psi.EnvironmentVariables['PYTHONIOENCODING'] = 'utf-8'
    $psi.EnvironmentVariables['PYTHONUTF8'] = '1'
    # Nothing from the calling shell may decide which code gets imported.
    $psi.EnvironmentVariables.Remove('VIRTUAL_ENV')
    $psi.EnvironmentVariables.Remove('PYTHONPATH')

    $p = [System.Diagnostics.Process]::Start($psi)
    $errTask = $p.StandardError.ReadToEndAsync()   # read both pipes at once,
    $stdout = $p.StandardOutput.ReadToEnd()        # or a full pipe deadlocks
    $p.WaitForExit()
    [pscustomobject]@{ Out = $stdout; Err = $errTask.Result; Code = $p.ExitCode }
}

# ---------------------------------------------------------------- masking
function Get-RootPattern([string]$Root) {
    # Matches the root with either slash direction; -replace ignores case.
    [regex]::Escape($Root.TrimEnd('\')) -replace '\\\\', '[\\/]'
}

function Get-Normalised([string]$Text) {
    $t = $Text -replace "`r`n", "`n"
    $t = $t -replace (Get-RootPattern $Trial), '<repo>'   # Trial first: Base
    $t = $t -replace (Get-RootPattern $Base), '<repo>'    # is its prefix
    $t = $t -replace '\b(?!00:00:00\b)\d{2}:\d{2}:\d{2}([.,]\d+)?\b', '<time>'
    $t = $t -replace '\bin \d+(\.\d+)?s( \(\d+:\d{2}:\d{2}\))?', 'in <dur>'
    $t = $t -replace '\b0x[0-9a-f]+\b', '<addr>'
    $t = $t -replace '\b(?=[0-9a-f]*[a-f])[0-9a-f]{16,}\b', '<hash>'
    $t = ($t -split "`n" | ForEach-Object { $_.TrimEnd() }) -join "`n"
    $t.TrimEnd()
}

# Which parts differ after masking: exit code, stdout, stderr. Empty = same.
function Get-Differences($B, $T) {
    $parts = @()
    if ($B.Code -ne $T.Code) { $parts += "exit code $($B.Code) vs $($T.Code)" }
    foreach ($s in 'Out', 'Err') {
        if (-not ((Get-Normalised $B.$s) -ceq (Get-Normalised $T.$s))) { $parts += $s.ToLower() }
    }
    , $parts
}

function Test-Network($B, $T) {
    ($B.Out + $B.Err + $T.Out + $T.Err) -match $NetworkPattern
}

# ---------------------------------------------------------------- jobs
$pyBase = Join-Path $Base '.venv\Scripts\python.exe'
$pyTrial = Join-Path $Trial '.venv\Scripts\python.exe'
foreach ($pair in @(@('base', $pyBase, $Base), @('trial', $pyTrial, $Trial))) {
    if (-not (Test-Path $pair[1])) { throw "[$($pair[0])] no venv python at $($pair[1])" }
    $r = Invoke-Captured $pair[1] @('-c', 'import moirai; print(moirai.__file__)') $pair[2]
    $imported = $r.Out.Trim()
    if (-not $imported.StartsWith($pair[2], [StringComparison]::OrdinalIgnoreCase)) {
        throw "[$($pair[0])] imports moirai from $imported, not from $($pair[2])"
    }
    Write-Host "[$($pair[0])] moirai -> $imported"
}

# Each job: a name and the arguments, relative to the repository root.
$jobs = New-Object System.Collections.Generic.List[object]
if (-not $NoPytest) { $jobs.Add(@('_pytest', @('-m', 'pytest', '-p', 'no:cacheprovider'))) }
$scriptNames = @(Get-ChildItem (Join-Path $Base 'scripts') -Filter '*.py' | ForEach-Object Name) +
               @(Get-ChildItem (Join-Path $Trial 'scripts') -Filter '*.py' | ForEach-Object Name) |
               Sort-Object -Unique
foreach ($n in $scriptNames) {
    if ($Skip -contains $n) { Write-Host "skip $n"; continue }
    $jobs.Add(@([System.IO.Path]::GetFileNameWithoutExtension($n), @("scripts\$n")))
}

# ---------------------------------------------------------------- run, interleaved
$real = New-Object System.Collections.Generic.List[string]
$inconclusive = New-Object System.Collections.Generic.List[string]
$pytestLine = @{}

foreach ($job in $jobs) {
    $name = $job[0]; $argList = $job[1]
    $inBase = ($name -eq '_pytest') -or (Test-Path (Join-Path $Base $argList[0]))
    $inTrial = ($name -eq '_pytest') -or (Test-Path (Join-Path $Trial $argList[0]))
    if (-not $inBase) { $real.Add("$name  (only in trial)"); continue }
    if (-not $inTrial) { $real.Add("$name  (only in base)"); continue }

    Write-Host "$name"
    $b = Invoke-Captured $pyBase $argList $Base
    $t = Invoke-Captured $pyTrial $argList $Trial
    $diffs = Get-Differences $b $t

    if ($diffs.Count -gt 0 -and (Test-Network $b $t)) {
        Write-Host "  network error on one side; retrying both in $RetryPauseSeconds s"
        Start-Sleep -Seconds $RetryPauseSeconds
        $b = Invoke-Captured $pyBase $argList $Base
        $t = Invoke-Captured $pyTrial $argList $Trial
        $diffs = Get-Differences $b $t
    }

    # Raw output, both sides
    foreach ($side in @(@('base', $b), @('trial', $t))) {
        $dir = Join-Path $Out $side[0]
        [System.IO.File]::WriteAllText((Join-Path $dir "$name.out.txt"), $side[1].Out, $Utf8)
        [System.IO.File]::WriteAllText((Join-Path $dir "$name.err.txt"), $side[1].Err, $Utf8)
    }
    if ($name -eq '_pytest') {
        $pytestLine['base'] = (($b.Out.Trim() -split "`n")[-1].Trim()) + "   (exit $($b.Code))"
        $pytestLine['trial'] = (($t.Out.Trim() -split "`n")[-1].Trim()) + "   (exit $($t.Code))"
    }
    if ($diffs.Count -eq 0) { continue }

    # Masked text and a diff for each stream that differs
    foreach ($s in 'Out', 'Err') {
        $nb = Get-Normalised $b.$s; $nt = Get-Normalised $t.$s
        if ($nb -ceq $nt) { continue }
        $low = $s.ToLower()
        $fb = Join-Path $Out "norm\base\$name.$low.txt"
        $ft = Join-Path $Out "norm\trial\$name.$low.txt"
        [System.IO.File]::WriteAllText($fb, $nb + "`n", $Utf8)
        [System.IO.File]::WriteAllText($ft, $nt + "`n", $Utf8)
        $d = @(& git diff --no-index --no-color -- $fb $ft)
        [System.IO.File]::WriteAllLines((Join-Path $Out "diff\$name.$low.diff"), [string[]]$d, $Utf8)
    }

    $label = "$name  (" + ($diffs -join ', ') + ")"
    if (Test-Network $b $t) { $inconclusive.Add($label) } else { $real.Add($label) }
}

# ---------------------------------------------------------------- summary
Write-Host ''
Write-Host '==================== SUMMARY ===================='
Write-Host "base : $Base"
Write-Host "trial: $Trial"
if (-not $NoPytest) {
    Write-Host "base  pytest: $($pytestLine['base'])"
    Write-Host "trial pytest: $($pytestLine['trial'])"
}
Write-Host "compared: $($jobs.Count) outputs, interleaved, stdout and stderr separately"
Write-Host ''
if ($inconclusive.Count -gt 0) {
    Write-Host "INCONCLUSIVE (network error after one retry): $($inconclusive.Count)" -ForegroundColor Yellow
    $inconclusive | ForEach-Object { Write-Host "  $_" }
}
if ($real.Count -eq 0) {
    Write-Host 'No real differences after masking.' -ForegroundColor Green
} else {
    Write-Host "Real differences after masking: $($real.Count)" -ForegroundColor Red
    $real | ForEach-Object { Write-Host "  $_" }
}
if ($real.Count + $inconclusive.Count -gt 0) {
    Write-Host "Diffs of the masked text: $(Join-Path $Out 'diff')"
}
exit ($real.Count + $inconclusive.Count)
