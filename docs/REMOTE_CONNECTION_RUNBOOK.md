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

```te
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
└── atlas21_training_raw/
    ├── isles26_5fold_split_test.json
    ├── isles26_nested_15_5_best.json
    ├── isles26_stratified_15pct.json
    └── Training_Raw/
```

---

## 7. Synchronize the dataset from home to LRZ

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

## 8. Troubleshooting

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

## 9. Security and operating rules

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

## 10. Quick operational checklist

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
