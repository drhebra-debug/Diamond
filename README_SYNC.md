# Auto-sync (git) setup for DIAMOND

This repository includes a simple PowerShell script to help keep the local workspace and remote git repository in sync.

Files
- `scripts/git_sync.ps1` — PowerShell script that:
  - `git fetch` and `git pull --rebase` from the configured remote/branch
  - Detects local changes and commits them with an automatic message
  - Pushes commits back to the remote branch
  - Logs activity to `git_sync.log` in the repository root

Security & credentials
- The script requires that `git push` can run non-interactively. Configure one of:
  - SSH key for your git host (add public key to remote account)
  - Git Credential Manager / cached credentials
- Optionally set `GIT_USER_NAME` and `GIT_USER_EMAIL` environment variables for auto-commits.

Usage examples

Dry-run (no commit/push):

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\git_sync.ps1 -RepoPath "C:\path\to\DIAMOND" -Branch main -DryRun
```

Run for real:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\git_sync.ps1 -RepoPath "C:\path\to\DIAMOND" -Branch main
```

Scheduling (Windows Task Scheduler)

1. Open Task Scheduler → Create Task.
2. Set a trigger (e.g., every 5 minutes or on logon).
3. Action: Start a program:
   - Program/script: `powershell`
   - Arguments: `-ExecutionPolicy Bypass -File "C:\path\to\DIAMOND\scripts\git_sync.ps1" -RepoPath "C:\path\to\DIAMOND" -Branch main`
4. Run whether user is logged on or not (store credentials) for background operation.

Notes & caveats
- Automatic commits will include any local file changes; consider filtering or adding a `.gitignore` for files you don't want synced.
- If `git pull --rebase` fails due to conflicts, the script attempts a fallback pull and logs the failure. Manual resolution may be required.
- The assistant cannot push on your behalf unless the host has valid git credentials configured.

If you want, I can also:
- Add a small Python wrapper to run cross-platform
- Create a GitHub Actions workflow to mirror commits from another remote
- Add a Windows Task Scheduler XML export for easy import

GitHub Actions mirror
---------------------

A workflow has been added at `.github/workflows/mirror.yml` that will push any repository changes to a remote mirror URL. To use it, add a repository secret named `MIRROR_REMOTE` containing the target remote git URL (for example `https://<token>@github.com/owner/repo.git`). Optionally set `GIT_MIRROR_NAME` and `GIT_MIRROR_EMAIL` secrets to configure the author for automated pushes.

Linux / macOS script
---------------------

We've also added a POSIX shell script: `scripts/git_sync.sh` which provides the same fetch/pull/commit/push behavior on Linux or macOS. Example usage:

```bash
./scripts/git_sync.sh /path/to/DIAMOND main origin
```

It writes actions to `git_sync.log` in the repository root.
