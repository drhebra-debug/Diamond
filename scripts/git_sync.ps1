<#
Auto Git sync script for DIAMOND repository (Windows PowerShell)

Usage examples:
  # Dry-run to see actions
  powershell -ExecutionPolicy Bypass -File .\scripts\git_sync.ps1 -RepoPath "C:\path\to\repo" -Branch main -DryRun

  # Real run (ensure SSH or credential helper configured)
  powershell -ExecutionPolicy Bypass -File .\scripts\git_sync.ps1 -RepoPath "C:\path\to\repo" -Branch main

Notes:
- Configure your git credentials (SSH key or credential helper) so `git push` can succeed non-interactively.
#>

param(
    [string]$RepoPath = (Split-Path -Path $PSScriptRoot -Parent),
    [string]$Branch = "main",
    [string]$Remote = "origin",
    [switch]$DryRun
)

Set-StrictMode -Version Latest
$log = Join-Path $RepoPath "git_sync.log"
function Log($msg){
    $t = (Get-Date).ToString('o')
    "$t - $msg" | Out-File -FilePath $log -Append -Encoding utf8
}

Push-Location $RepoPath
try {
    Log "Starting sync. RepoPath=$RepoPath Branch=$Branch Remote=$Remote DryRun=$DryRun"

    # Fetch remote refs
    $fetch = git fetch $Remote 2>&1
    if ($LASTEXITCODE -ne 0) { Log "git fetch failed: $fetch" }

    # Try rebase pull to keep history linear
    $pull = git pull --rebase $Remote $Branch 2>&1
    if ($LASTEXITCODE -ne 0) {
        Log "git pull --rebase failed: $pull"
        try { git rebase --abort 2>$null } catch {}
        # Fallback to a normal pull
        $pull2 = git pull $Remote $Branch 2>&1
        if ($LASTEXITCODE -ne 0) { Log "Fallback git pull failed: $pull2" }
        else { Log "Fallback git pull succeeded: $pull2" }
    } else {
        Log "git pull --rebase succeeded: $pull"
    }

    # Detect local changes
    $status = git status --porcelain
    if (-not [string]::IsNullOrEmpty($status)) {
        Log "Local changes detected:\n$status"
        if ($DryRun) {
            Log "DryRun enabled — skipping commit/push"
        } else {
            git add -A
            # Use provided env vars if available for commit identity
            $nameOpt = if ($env:GIT_USER_NAME) { "-c user.name=\"$env:GIT_USER_NAME\"" } else { "" }
            $emailOpt = if ($env:GIT_USER_EMAIL) { "-c user.email=\"$env:GIT_USER_EMAIL\"" } else { "" }
            $commitCmd = "git $nameOpt $emailOpt commit -m \"Auto-sync: $(Get-Date -Format o)\""
            try {
                iex $commitCmd | Out-Null
                Log "Committed local changes"
            } catch {
                Log "Commit failed or nothing to commit"
            }

            # Push to the tracked branch
            $push = git push $Remote HEAD:$Branch 2>&1
            if ($LASTEXITCODE -ne 0) { Log "git push failed: $push" }
            else { Log "git push succeeded: $push" }
        }
    } else {
        Log "No local changes to commit."
    }

} catch {
    Log "Exception during sync: $_"
} finally {
    Pop-Location
}

Exit 0
