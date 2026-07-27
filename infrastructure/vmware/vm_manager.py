"""
infrastructure/vmware/vm_manager.py
-------------------------------------
Python 3 faithful rewrite of vm.py VMware commands.

Original vm.py used VBoxManage. This replaces every VBoxManage
call with the VMware equivalent using vmrun + ovftool.

VBoxManage → VMware mapping (line by line from vm.py):
  vboxmanage import <ova>               → ovftool <ova> <dest_dir>/
  VBoxManage list vms                   → vmrun list  (all registered)
  VBoxManage list runningvms            → vmrun list  (running only)
  VBoxManage modifyvm --name            → edit VMX displayName
  VBoxManage modifyvm --nic1 bridged    → edit VMX ethernet0.connectionType
  VBoxManage startvm --type headless    → vmrun start <vmx> nogui

Requires on host:
  vmrun   (VMware Workstation → /usr/lib/vmware/bin/vmrun)
  ovftool (VMware OVF Tool   → /usr/bin/ovftool)
"""

from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

VM_STORE_PATH   = os.environ.get("FMRAG_VM_STORE",    "/var/fmrag/vms")
OVA_UPLOAD_PATH = os.environ.get("FMRAG_OVA_UPLOAD",  "/var/www/html/uploads")
VMRUN_BIN       = os.environ.get("VMRUN_BIN",          "vmrun")
OVFTOOL_BIN     = os.environ.get("OVFTOOL_BIN",        "ovftool")
BRIDGE_IFACE    = os.environ.get("FMRAG_BRIDGE_IFACE", "ens33")
VMWARE_TYPE     = os.environ.get("VMWARE_TYPE",        "ws")


# ── helpers ───────────────────────────────────────────────────────────────────

def _run(cmd: list) -> int:
    """Run a command, return exit code. Mirrors os.system() from vm.py."""
    log.debug("$ %s", " ".join(str(c) for c in cmd))
    return subprocess.run(cmd, capture_output=True).returncode


def _out(cmd: list) -> str:
    """Run a command, return stdout. Mirrors os.popen().read() from vm.py."""
    return subprocess.run(cmd, capture_output=True, text=True).stdout


def _vmrun(*args) -> int:
    return _run([VMRUN_BIN, "-T", VMWARE_TYPE, *args])


def _vmrun_out(*args) -> str:
    return _out([VMRUN_BIN, "-T", VMWARE_TYPE, *args])


# ── VMX helpers ───────────────────────────────────────────────────────────────

def _find_vmx(vm_name: str) -> Optional[str]:
    """Return .vmx path for a registered VM name, or None."""
    base = Path(VM_STORE_PATH) / vm_name
    if base.exists():
        vmx_files = list(base.glob("*.vmx"))
        if vmx_files:
            return str(vmx_files[0])
    return None


def _read_vmx(vmx: str) -> dict:
    settings = {}
    with open(vmx) as f:
        for line in f:
            if "=" in line:
                k, _, v = line.partition("=")
                settings[k.strip()] = v.strip().strip('"')
    return settings


def _write_vmx(vmx: str, settings: dict):
    lines = [f'{k} = "{v}"' for k, v in settings.items()]
    with open(vmx, "w") as f:
        f.write("\n".join(lines) + "\n")


# ── list vms (replaces VBoxManage list vms / list runningvms) ────────────────

def list_vms() -> str:
    """
    Returns a string of registered VM names.
    Mirrors: os.popen('VBoxManage list vms').read()
    vmrun list shows only RUNNING vms; we scan VM_STORE_PATH for all.
    """
    names = []
    base = Path(VM_STORE_PATH)
    if base.exists():
        for d in base.iterdir():
            if d.is_dir() and list(d.glob("*.vmx")):
                names.append(d.name)
    return "\n".join(f'"{n}"' for n in names)


def list_running_vms() -> str:
    """
    Returns running VM names.
    Mirrors: os.popen('VBoxManage list runningvms').read()
    """
    return _vmrun_out("list")


def vm_in_list(vm_name: str, vm_list_str: str) -> bool:
    """Mirrors: if vm_name in vm_list"""
    return vm_name in vm_list_str


# ── import OVA (replaces vboxmanage import) ───────────────────────────────────

def import_ova(vm_file_name: str, vm_name: str) -> int:
    """
    Import an OVA into VMware format.
    Mirrors: os.system('cd /var/www/html/uploads && vboxmanage import %s' % vm_file_name)
    Returns 0 on success, non-zero on failure.
    """
    ova_path = os.path.join(OVA_UPLOAD_PATH, vm_file_name)
    if not os.path.isfile(ova_path):
        log.error("OVA not found: %s", ova_path)
        return 1

    dest_dir = os.path.join(VM_STORE_PATH, vm_name)
    os.makedirs(dest_dir, exist_ok=True)

    rc = _run([
        OVFTOOL_BIN,
        "--acceptAllEulas",
        f"--name={vm_name}",
        ova_path,
        dest_dir,
    ])
    if rc == 0:
        log.info("OVA imported: %s → %s", vm_file_name, dest_dir)
    else:
        log.error("OVA import failed: %s (rc=%d)", vm_file_name, rc)
    return rc


# ── rename VM (replaces VBoxManage modifyvm --name) ───────────────────────────

def rename_vm(old_name: str, new_name: str) -> int:
    """
    Rename VM by editing VMX displayName.
    Mirrors: os.system('VBoxManage modifyvm %s --name %s' % (old, new))
    Returns 0 on success.
    """
    vmx = _find_vmx(old_name)
    if not vmx:
        log.error("Cannot rename: VMX not found for %s", old_name)
        return 1

    settings = _read_vmx(vmx)
    settings["displayName"] = new_name
    _write_vmx(vmx, settings)

    # Rename directory
    old_dir = Path(vmx).parent
    new_dir = old_dir.parent / new_name
    old_dir.rename(new_dir)
    log.info("VM renamed: %s → %s", old_name, new_name)
    return 0


# ── set bridged NIC (replaces VBoxManage modifyvm --nic1 bridged) ─────────────

def set_bridged_nic(vm_name: str, host_iface: str = BRIDGE_IFACE) -> int:
    """
    Set NIC0 to bridged on host_iface.
    Mirrors: os.system('VBoxManage modifyvm %s --nic1 bridged --bridgeadapter1 ens33')
    Returns 0 on success.
    """
    vmx = _find_vmx(vm_name)
    if not vmx:
        log.error("Cannot set NIC: VMX not found for %s", vm_name)
        return 1

    settings = _read_vmx(vmx)
    settings["ethernet0.present"]        = "TRUE"
    settings["ethernet0.connectionType"] = "bridged"
    settings["ethernet0.bridgeDev"]      = host_iface
    settings["ethernet0.virtualDev"]     = "vmxnet3"
    settings["ethernet0.addressType"]    = "generated"
    _write_vmx(vmx, settings)
    log.info("Bridged NIC set on %s (iface=%s)", vm_name, host_iface)
    return 0


# ── start VM (replaces VBoxManage startvm --type headless) ───────────────────

def start_vm(vm_name: str) -> int:
    """
    Start a VM headlessly.
    Mirrors: os.system('VBoxManage startvm %s --type headless' % vm_name)
    Returns 0 on success.
    """
    vmx = _find_vmx(vm_name)
    if not vmx:
        log.error("Cannot start: VMX not found for %s", vm_name)
        return 1

    rc = _vmrun("start", vmx, "nogui")
    if rc == 0:
        log.info("VM started: %s", vm_name)
    else:
        log.error("VM start failed: %s (rc=%d)", vm_name, rc)
    return rc


# ── SCP migration (unchanged from vm.py — sshpass scp) ───────────────────────

def migrate_ova(vm_file_name: str, dst_ip: str,
                dst_user: str = None,
                dst_pass: str = None) -> int:
    """
    SCP the OVA file to the destination cloudlet.
    Mirrors exactly:
      sshpass -p Gift123 scp -r /var/www/html/uploads/<file> stack@<dst>:/var/www/html/uploads
    Returns 0 on success.
    """
    dst_user = dst_user or os.environ.get("FMRAG_VM_USER", "fmrag")
    dst_pass = dst_pass or os.environ.get("FMRAG_VM_PASS", "")
    src = os.path.join(OVA_UPLOAD_PATH, vm_file_name)

    if dst_pass:
        cmd = [
            "sshpass", f"-p{dst_pass}",
            "scp", "-o", "StrictHostKeyChecking=no", "-r",
            src,
            f"{dst_user}@{dst_ip}:{OVA_UPLOAD_PATH}",
        ]
    else:
        cmd = [
            "scp", "-o", "StrictHostKeyChecking=no", "-r",
            src,
            f"{dst_user}@{dst_ip}:{OVA_UPLOAD_PATH}",
        ]

    rc = _run(cmd)
    if rc == 0:
        log.info("OVA migrated: %s → %s", vm_file_name, dst_ip)
    else:
        log.error("OVA migration failed: %s → %s (rc=%d)", vm_file_name, dst_ip, rc)
    return rc
