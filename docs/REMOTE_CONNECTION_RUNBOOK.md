# Remote Connection Runbook: Home Desktop and LRZ

## Purpose

This runbook is the one-stop operational reference for reaching the project's two critical remote endpoints:

1. The Windows 10 home desktop and its WSL2 Ubuntu environment.
2. The LRZ AI Systems login host, reached from the home desktop's WSL environment.

It records the connection topology, verified commands, credential locations, dataset paths, synchronization procedure, and troubleshooting lessons established on 2026-07-28.

No password, private key, or public-key body belongs in this document or repository.

---

## 1. Connection topology

```text
Local Windows workstation
  |
  | SSH using ~/.ssh/home_desktop
  v
Home Windows 10 desktop
  100.104.89.125
  user: minanessiem
  host: DESKTOP-E5N0CSS
  |
  | wsl.exe -d Ubuntu
  v
Home WSL2 Ubuntu 22.04
  user: minanessiem
  |
  | SSH password authentication via sshpass
  v
LRZ AI Systems
  di38tap@login.ai.lrz.de
  observed login node: login-02 (may vary)
```

The LRZ connection is intentionally made from home WSL. LRZ sessions are not kept alive with `ControlMaster` or another persistence mechanism.

---

## 2. Endpoint inventory

### Home desktop

| Item | Value |
|---|---|
| Address | `100.104.89.125` |
| Windows SSH user | `minanessiem` |
| Verified hostname | `DESKTOP-E5N0CSS` |
| Local private key | `%USERPROFILE%\.ssh\home_desktop` |
| Key fingerprint | `SHA256:9tQmBGA8m2Ft5+q+LlfVlz/Z02qc4X5HCnv3EUNbcP4` |
| WSL distribution | `Ubuntu` |
| Verified WSL release | Ubuntu 22.04.3 LTS |
| WSL user | `minanessiem` |

The current `home_desktop` key was created without a passphrase. Protect the private key with local filesystem permissions and never copy it into the repository.

### LRZ

| Item | Value |
|---|---|
| SSH endpoint | `login.ai.lrz.de` |
| SSH user | `di38tap` |
| Authentication | Password via `sshpass` in home WSL |
| Password file | `/home/minanessiem/.config/lrz/password` |
| Storage variable | `$SSD_STORE` |
| Observed `$SSD_STORE` value | `/dss/dssmcmlfs01/pn76ge/pn76ge-dss-0000/di38tap/` |

The LRZ password file is plaintext. It must remain outside every repository and have mode `600`.

---

## 3. Connect to the home desktop

Run these commands from PowerShell on the local Windows workstation.

### Test the Windows SSH endpoint

```powershell
ssh -i "$env:USERPROFILE\.ssh\home_desktop" `
  -o IdentitiesOnly=yes `
  -o BatchMode=yes `
  -o ConnectTimeout=10 `
  minanessiem@100.104.89.125 `
  "hostname"
```

Expected output:

```text
DESKTOP-E5N0CSS
```

### Open an interactive Windows shell

```powershell
ssh -i "$env:USERPROFILE\.ssh\home_desktop" `
  -o IdentitiesOnly=yes `
  minanessiem@100.104.89.125
```

### Open home WSL2 Ubuntu directly

```powershell
ssh -tt -i "$env:USERPROFILE\.ssh\home_desktop" `
  -o IdentitiesOnly=yes `
  minanessiem@100.104.89.125 `
  "wsl.exe -d Ubuntu"
```

Use `-tt` for WSL commands when output is blank or console control is required. Windows OpenSSH and `wsl.exe` did not reliably return captured output without a pseudo-terminal during setup.

### Run one command inside home WSL

```powershell
ssh -tt -i "$env:USERPROFILE\.ssh\home_desktop" `
  -o IdentitiesOnly=yes `
  -o BatchMode=yes `
  minanessiem@100.104.89.125 `
  'wsl.exe -d Ubuntu -- bash -lc "pwd"'
```

For commands with complex quoting, prefer opening WSL interactively and running the Linux command there. Three nested parsers (local PowerShell, remote Windows command processing, and the Linux shell) make heavily quoted one-liners fragile.

### Activate the project environment in home WSL

Before running project Python scripts in the home WSL environment, activate the shared MedSegDiff virtual environment:

```bash
source /mnt/c/Users/minanessiem/Development/MedSegDiff_env/bin/activate
```

For a non-interactive command, activate the environment in the same `bash -lc` invocation before changing to the repository and running Python. Virtual-environment activation is shell-local and does not persist across SSH commands.

### Home WSL project workspace

The verified equivalent of this project on the home desktop is:

```text
/mnt/c/Users/minanessiem/Development/medseg-diffusion
```
Related project paths are:

| Purpose | Home WSL path |
|---|---|
| Project repository | `/mnt/c/Users/minanessiem/Development/medseg-diffusion` |
| Python environment | `/mnt/c/Users/minanessiem/Development/MedSegDiff_env` |
| ISLES26 dataset and split-output root | `/mnt/c/Users/minanessiem/Development/isles26_combined` |
| ISLES26 split creator, relative to the repository | `scripts/dataset_setup/ISLES26_json_creator.py` |

After connecting to home WSL, use this preflight before project work:

```bash
cd /mnt/c/Users/minanessiem/Development/medseg-diffusion
source /mnt/c/Users/minanessiem/Development/MedSegDiff_env/bin/activate
pwd
command -v python
git status --short
git branch --show-current
```

The branch and commit are intentionally not fixed in this runbook. Inspect them at the start of each session before editing files or running version-sensitive scripts.

For a short non-interactive check from the local Windows workstation:

```powershell
ssh -tt -i "$env:USERPROFILE\.ssh\home_desktop" `
  -o IdentitiesOnly=yes `
  -o BatchMode=yes `
  minanessiem@100.104.89.125 `
  'wsl.exe -d Ubuntu -- bash -lc "cd /mnt/c/Users/minanessiem/Development/medseg-diffusion && source /mnt/c/Users/minanessiem/Development/MedSegDiff_env/bin/activate && pwd && command -v python && git status --short"'
```

For longer automation, prefer an interactive WSL session or send an LF-terminated Bash script without a UTF-8 BOM. Piping multiline text from Windows PowerShell through Windows OpenSSH can introduce a BOM or CRLF line endings; Bash may then misread the first command or retain a trailing carriage return in paths. Automation scripts that create outputs should also use `set -euo pipefail`, write to a temporary file, validate it, and publish it with an explicit no-overwrite check.

### Git workflow across the laptop and home desktop

The local laptop's GitHub-approved SSH key is in the laptop WSL2 environment, not in Windows OpenSSH:

| Item | Value |
|---|---|
| Laptop WSL user | `nathanail` |
| Laptop WSL key | `/home/nathanail/.ssh/id_ed25519` |
| Approved fingerprint | `SHA256:Dbreg/tw8d/u0p3hlHIEdQccLThqMCbkIYJ7kRhsZG4` |
| Laptop repository in WSL | `/mnt/c/Users/konst/Development/medseg-diffusion_ISLES24` |

Windows OpenSSH uses `~/.ssh/home_desktop` to reach the home desktop. It does not automatically use the GitHub key stored inside laptop WSL. Therefore, run GitHub fetch/push operations for the laptop checkout through WSL:

```powershell
wsl.exe -- git -C /mnt/c/Users/konst/Development/medseg-diffusion_ISLES24 status --short --branch
wsl.exe -- git -C /mnt/c/Users/konst/Development/medseg-diffusion_ISLES24 push origin HEAD
```

Before updating the home checkout, inspect its branch and working tree. Never assume it is clean or synchronized merely because its local remote-tracking display looks current:

```bash
cd /mnt/c/Users/minanessiem/Development/medseg-diffusion
git status --short --branch
git fetch origin
git log --oneline --left-right HEAD...origin/$(git branch --show-current)
git diff --name-only HEAD..origin/$(git branch --show-current)
```

Use a fast-forward-only pull so Git cannot create an implicit merge commit:

```bash
git pull --ff-only origin "$(git branch --show-current)"
```

If incoming paths overlap unpublished home changes, stop and inspect both diffs. Do not stash the entire worktree, especially when it contains large untracked datasets, runs, or generated files. When explicitly appropriate, stash only the overlapping tracked path, pull, and immediately restore it:

```bash
git stash push -m scoped-pull-preservation -- path/to/overlapping_file.py
git pull --ff-only origin "$(git branch --show-current)"
git stash pop stash@{0}
```

If the stash reapplication conflicts, leave the retained stash in place and resolve deliberately. Never discard or broadly reset home worktree changes to make a pull succeed.

Home WSL may occasionally fail to resolve `github.com`. Diagnose first:

```bash
getent hosts github.com
cat /etc/resolv.conf
```

Do not record a GitHub IP address permanently; GitHub addresses can change. If an urgent one-command workaround is necessary, resolve a fresh address using trusted DNS on a working machine and override only that Git invocation while retaining `github.com` for host-key verification:

```bash
git -c 'core.sshCommand=ssh -o HostName=<fresh-github-ip> -o HostKeyAlias=github.com' \
  pull --ff-only origin "$(git branch --show-current)"
```

---

## 4. Home desktop key setup and recovery

This section is for rebuilding access, not routine use.

### Create the local key

```powershell
ssh-keygen -t ed25519 -a 100 `
  -f "$env:USERPROFILE\.ssh\home_desktop" `
  -C "home desktop SSH access"
```

The public key is the complete single line in:

```powershell
Get-Content "$env:USERPROFILE\.ssh\home_desktop.pub"
```

An authorized key begins with `ssh-ed25519`, followed by a long base64 value and an optional comment. A `SHA256:...` fingerprint is only an identifier; it is not an authorized key.

### Windows administrator account key location

Because `minanessiem` is handled as an administrator by Windows OpenSSH, the working authorized-key location is:

```text
C:\ProgramData\ssh\administrators_authorized_keys
```

From an elevated PowerShell on the home desktop, apply strict ACLs:

```powershell
icacls "$env:ProgramData\ssh\administrators_authorized_keys" `
  /inheritance:r `
  /grant:r "*S-1-5-32-544:F" `
  /grant "*S-1-5-18:F"
```

The SIDs identify the built-in Administrators group and SYSTEM and work independently of the Windows display language.

Verify the installed public key:

```powershell
ssh-keygen -lf "$env:ProgramData\ssh\administrators_authorized_keys"
```

Expected fingerprint:

```text
SHA256:9tQmBGA8m2Ft5+q+LlfVlz/Z02qc4X5HCnv3EUNbcP4
```

---

## 5. Connect to LRZ through home WSL

### Prerequisites inside home WSL

```bash
command -v sshpass
stat -c '%a %U %n' ~/.config/lrz/password
```

Expected password-file state:

```text
600 minanessiem /home/minanessiem/.config/lrz/password
```

If `sshpass` must be installed:

```bash
sudo apt update
sudo apt install -y sshpass
```

Create or replace the password file without putting the password in shell history:

```bash
install -d -m 700 ~/.config/lrz
read -rsp "LRZ password: " LRZ_PASSWORD
printf '\n'
printf '%s\n' "$LRZ_PASSWORD" > ~/.config/lrz/password
unset LRZ_PASSWORD
chmod 600 ~/.config/lrz/password
```

### Test LRZ from inside home WSL

```bash
sshpass -f ~/.config/lrz/password \
  ssh \
  -o PreferredAuthentications=password \
  -o PubkeyAuthentication=no \
  -o StrictHostKeyChecking=accept-new \
  -o ConnectTimeout=15 \
  di38tap@login.ai.lrz.de \
  hostname
```

The resolved login node can change. `login-02` was observed during setup but must not be hard-coded.

### Test LRZ end-to-end from the local Windows workstation

```powershell
ssh -tt -i "$env:USERPROFILE\.ssh\home_desktop" `
  -o IdentitiesOnly=yes `
  -o BatchMode=yes `
  -o ConnectTimeout=10 `
  minanessiem@100.104.89.125 `
  'wsl.exe -d Ubuntu -- sshpass -f /home/minanessiem/.config/lrz/password ssh -o PreferredAuthentications=password -o PubkeyAuthentication=no -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15 di38tap@login.ai.lrz.de hostname'
```

### Open an interactive LRZ shell from the local workstation

```powershell
ssh -tt -i "$env:USERPROFILE\.ssh\home_desktop" `
  -o IdentitiesOnly=yes `
  minanessiem@100.104.89.125 `
  'wsl.exe -d Ubuntu -- sshpass -f /home/minanessiem/.config/lrz/password ssh -tt -o PreferredAuthentications=password -o PubkeyAuthentication=no di38tap@login.ai.lrz.de'
```

Exit the LRZ shell when work is complete. This workflow does not create a persistent LRZ control connection.

---

## 6. Critical dataset paths

### Home WSL source

```text
/mnt/c/Users/minanessiem/Development/isles26_combined
```

### LRZ destination

```text
$SSD_STORE/datasets/isles26_combined
```

Observed resolved path:

```text
/dss/dssmcmlfs01/pn76ge/pn76ge-dss-0000/di38tap/datasets/isles26_combined
```

Prefer `$SSD_STORE` in an interactive LRZ shell. For deeply nested non-interactive commands, resolving it first with `printenv SSD_STORE` and then using the absolute path avoids expansion at the wrong SSH hop.

The expected dataset root is:

```text
isles26_combined/
`-- atlas21_training_raw/
    |-- isles26_5fold_split_test.json
    |-- isles26_nested_15_5_best.json
    |-- isles26_nested_15_5_best_2026-07-28.json
    |-- isles26_stratified_15pct.json
    `-- Training_Raw/
```

---

## 7. Regenerate an ISLES26 split safely

Run split generation from home WSL after pulling the intended repository revision.

### Stable paths and environment

```bash
cd /mnt/c/Users/minanessiem/Development/medseg-diffusion
source /mnt/c/Users/minanessiem/Development/MedSegDiff_env/bin/activate

dataset=/mnt/c/Users/minanessiem/Development/isles26_combined
split_dir="$dataset/atlas21_training_raw"
```

The dataset root is the split creator's input. Final split JSON files belong in `atlas21_training_raw` next to the older split files, not at the top level of `isles26_combined`.

### Preflight

Confirm the repository revision, environment, script interface, dataset case discovery, and intended filename before writing anything:

```bash
pwd
command -v python
git status --short --branch
git log -1 --oneline
python scripts/dataset_setup/ISLES26_json_creator.py --help
find "$split_dir" -maxdepth 1 -type f -name '*.json' -print | sort
```

Use an ISO date in the filename, for example `isles26_nested_15_5_best_YYYY-MM-DD.json`. Refuse to continue when that exact target already exists.

### Test and generate through a temporary file

The standard nested search uses 15% full validation and 5% fast validation. `val_full` is the union of `val_rest` and `val_fast`.

In the command below, 100 outer seeds and 100 inner seeds mean 10,000 nested candidate combinations, not 100 total candidates. On 1,284 cases this took approximately 13 minutes on the home desktop.

```bash
set -euo pipefail

python -m unittest tests.test_isles26_json_creator -v

date_tag=$(date +%F)
target="$split_dir/isles26_nested_15_5_best_${date_tag}.json"
if [[ -e "$target" ]]; then
  printf 'Refusing to overwrite existing split: %s\n' "$target" >&2
  exit 1
fi

tmp=$(mktemp "$split_dir/.isles26_nested_15_5_best_${date_tag}.json.tmp.XXXXXX")
trap 'rm -f "$tmp"' EXIT

python scripts/dataset_setup/ISLES26_json_creator.py \
  "$dataset" \
  "$tmp" \
  --val_full_size 15 \
  --val_fast_size 5 \
  --seed 42 \
  --num_outer_split_seeds 100 \
  --num_inner_split_seeds 100
```

Do not publish the temporary JSON until all of the following have been checked:

- The JSON parses successfully and contains a `training` list and `validation_policy` object.
- Every `caseID` is unique.
- The only split labels are `train`, `val_rest`, and `val_fast`.
- `val_rest + val_fast` equals the rounded 15% `val_full` target.
- `val_fast` equals the rounded 5% target.
- The policy records 100 outer and 100 inner candidates.
- Every singleton site's only case is assigned to training.
- `singleton_training_sites` matches the singleton sites derived from the case metadata.
- Every multi-case site retains at least one training and one full-validation case.

After validation, publish without an overwrite race and record the checksum:

```bash
ln "$tmp" "$target"
rm "$tmp"
trap - EXIT
sha256sum "$target"
```

Hard-link creation is atomic on this dataset filesystem and fails if the target appeared after the initial existence check. A generation failure leaves only the temporary file, which the shell trap removes.

### Split-assignment policy

The outer split assigns proportional site quotas. Singleton sites receive a validation quota of zero and remain in training. Every site with two or more cases receives at least one full-validation case and retains at least one training case.

Within each site, selection is spread across available days-post-stroke and chronicity strata. The inner `val_fast` subset uses proportional site quotas but does not require every site to appear because it is much smaller.

Candidate quality is scored using the summed absolute percentage-point differences across:

- Days-post-stroke bins
- Chronicity
- ATLAS2 dataset/source membership

Site balance is enforced through quotas and coverage constraints rather than included directly in this numerical score. The total nested score is:

```text
distance(val_full, all) + distance(val_fast, val_full) + distance(val_fast, all)
```

Lower is better.

### Generation record: 2026-07-28

| Item | Result |
|---|---|
| Dataset cases | 1,284 |
| Train | 1,091 (84.97%) |
| `val_rest` | 129 (10.05%) |
| `val_fast` | 64 (4.98%) |
| `val_full` | 193 (15.03%) |
| Singleton training-only sites | `R016`, `R020`, `R063` |
| Base seed | `42` |
| Selected outer seed | `3813457838` |
| Selected inner seed | `2920397057` |
| Outer balance score | `10.50` |
| Inner score, `val_fast` vs `val_full` | `15.77` |
| Inner score, `val_fast` vs all | `17.39` |
| Total nested score | `43.66` |
| Output file | `atlas21_training_raw/isles26_nested_15_5_best_2026-07-28.json` |
| SHA-256 | `349f890a1d64e578b7ac258668d903a4b7861899cf64ab6f215b62d85825576b` |

This table is a dated operational record, not a permanent expected result. Recompute and record counts, selected seeds, scores, singleton sites, and checksum whenever the dataset changes.

---

## 8. Synchronize the dataset from home to LRZ

Run `rsync` from home WSL. The direction is always:

```text
home desktop source -> LRZ destination
```

### Why checksum comparison is required

The Windows-mounted source and LRZ copy have widespread modification-time differences. A normal size-and-time comparison incorrectly proposed about 3.09 GB across 2,077 files.

Checksum comparison correctly isolated new or content-changed files. On 2026-07-28 it found and successfully transferred:

- 1,146 regular files,
- 223 new files,
- 225 new directories,
- 151.84 MB of content,
- zero deletions.

Checksum comparison reads both copies and is slower, but prevents retransmitting identical NIfTI files merely because their timestamps differ.

### Mandatory dry-run

From home WSL:

```bash
sshpass -f ~/.config/lrz/password \
  rsync -rcn \
  --exclude=.DS_Store \
  --stats \
  --human-readable \
  /mnt/c/Users/minanessiem/Development/isles26_combined/ \
  di38tap@login.ai.lrz.de:/dss/dssmcmlfs01/pn76ge/pn76ge-dss-0000/di38tap/datasets/isles26_combined/
```

Review at least:

- `Number of created files`
- `Number of deleted files` (must be zero for this workflow)
- `Number of regular files transferred`
- `Total transferred file size`

Add `-i` to `rsync` for an itemized filename diff:

```bash
sshpass -f ~/.config/lrz/password \
  rsync -rcni \
  --exclude=.DS_Store \
  /mnt/c/Users/minanessiem/Development/isles26_combined/ \
  di38tap@login.ai.lrz.de:/dss/dssmcmlfs01/pn76ge/pn76ge-dss-0000/di38tap/datasets/isles26_combined/
```

### Perform the transfer

```bash
sshpass -f ~/.config/lrz/password \
  rsync -rc \
  --exclude=.DS_Store \
  --partial \
  --stats \
  --human-readable \
  /mnt/c/Users/minanessiem/Development/isles26_combined/ \
  di38tap@login.ai.lrz.de:/dss/dssmcmlfs01/pn76ge/pn76ge-dss-0000/di38tap/datasets/isles26_combined/
```

Important semantics:

- `-r`: recurse through the dataset.
- `-c`: determine changes using content checksums rather than timestamps.
- `--partial`: retain partially transferred files if interrupted.
- `--exclude=.DS_Store`: do not copy macOS metadata files.
- The trailing `/` on both paths means synchronize the contents of the named directory.
- There is deliberately no `--delete`; LRZ-only files are preserved.
- There is deliberately no `-t`; timestamp-only differences are ignored.
- Do not use `--ignore-existing` when updated existing files must also be synchronized.

Run the dry-run again after transfer if an explicit zero-diff verification is required.

---

## 9. Troubleshooting

### Home connection fails at port 22 with `Permission denied`

Observed cause: NordVPN interfered with the route to `100.104.89.125`.

Action:

1. Disconnect NordVPN.
2. Confirm the network path used for `100.104.89.125` is active.
3. Retry the home hostname test.

This error occurs before SSH authentication and is distinct from `Permission denied (publickey,...)`.

### Home reports `Permission denied (publickey,password,keyboard-interactive)`

Check that:

1. The client is explicitly using `~/.ssh/home_desktop`.
2. `IdentitiesOnly=yes` is set while diagnosing.
3. `administrators_authorized_keys` contains the full `ssh-ed25519 ...` public-key line, not the SHA256 fingerprint.
4. The Windows ACLs match the commands in Section 4.
5. The installed-key fingerprint matches the expected fingerprint.

For a focused client-side probe:

```powershell
ssh -vvv `
  -i "$env:USERPROFILE\.ssh\home_desktop" `
  -o IdentitiesOnly=yes `
  -o BatchMode=yes `
  minanessiem@100.104.89.125 `
  "hostname"
```

Look for `Offering public key` and `Server accepts key`. Never paste verbose logs publicly without checking them for sensitive host information.

### WSL command succeeds but prints nothing

Allocate a pseudo-terminal using `ssh -tt`. This was required for reliable WSL console output through Windows OpenSSH.

### LRZ reports `Permission denied (publickey,password)`

Check inside home WSL:

```bash
command -v sshpass
stat -c '%a %U %n' ~/.config/lrz/password
```

Then replace the password file interactively if the LRZ password changed. Do not print the password or pass it as a command-line argument.

### LRZ host key warning

Do not bypass an unexpected host-key change. Confirm the current LRZ host-key fingerprint through an authoritative LRZ channel before updating `known_hosts`.

### `rsync` proposes an unexpectedly large transfer

Stop and verify that:

1. `-c` is present.
2. The source and destination directions are correct.
3. Both paths include the intended trailing `/`.
4. `--delete` is absent.
5. The dry-run uses `-n`.

Use `-i` to distinguish new files (`<f+++++++++`) from content changes (`<fc...`). Timestamp-only changes should not appear in the `-rcn` plan.

---

## 10. Security and operating rules

1. Never commit or paste the home private key.
2. Never commit, print, or paste the LRZ password file.
3. Keep `~/.config/lrz` at mode `700` and its password file at mode `600`.
4. Remember that `sshpass` makes unattended LRZ access possible for any process running as the same home WSL user.
5. Do not create persistent LRZ SSH control sessions.
6. Use `BatchMode=yes` for home automation so commands fail rather than unexpectedly prompt.
7. Use an `rsync` dry-run before every material synchronization.
8. Do not add `--delete` without separately reviewing and authorizing the deletion plan.
9. Treat the LRZ login hostname as a gateway; do not hard-code a particular resolved node such as `login-02`.
10. Close interactive sessions when work is complete.

---

## 11. Quick operational checklist

### Reach home

```powershell
ssh -i "$env:USERPROFILE\.ssh\home_desktop" -o IdentitiesOnly=yes minanessiem@100.104.89.125 "hostname"
```

### Reach home WSL

```powershell
ssh -tt -i "$env:USERPROFILE\.ssh\home_desktop" -o IdentitiesOnly=yes minanessiem@100.104.89.125 "wsl.exe -d Ubuntu"
```

### Reach LRZ from home WSL

```bash
sshpass -f ~/.config/lrz/password ssh -o PreferredAuthentications=password -o PubkeyAuthentication=no di38tap@login.ai.lrz.de
```

### Regenerate an ISLES26 split

```bash
cd /mnt/c/Users/minanessiem/Development/medseg-diffusion
source /mnt/c/Users/minanessiem/Development/MedSegDiff_env/bin/activate
git status --short --branch
python -m unittest tests.test_isles26_json_creator -v
# Follow Section 7: generate to a temporary file, validate, then publish without overwrite.
```

### Sync dataset safely

```bash
# First: dry-run
sshpass -f ~/.config/lrz/password rsync -rcn --exclude=.DS_Store --stats --human-readable \
  /mnt/c/Users/minanessiem/Development/isles26_combined/ \
  di38tap@login.ai.lrz.de:/dss/dssmcmlfs01/pn76ge/pn76ge-dss-0000/di38tap/datasets/isles26_combined/

# Then: transfer after reviewing the dry-run
sshpass -f ~/.config/lrz/password rsync -rc --exclude=.DS_Store --partial --stats --human-readable \
  /mnt/c/Users/minanessiem/Development/isles26_combined/ \
  di38tap@login.ai.lrz.de:/dss/dssmcmlfs01/pn76ge/pn76ge-dss-0000/di38tap/datasets/isles26_combined/
```
