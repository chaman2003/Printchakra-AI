"""
PrintChakra Backend - Printer Service

Service for printer management, queue operations, and diagnostics.
"""

import os
import sys
import json
import subprocess
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


class PrinterService:
    """Service for printer management and queue operations."""
    
    # Default print configuration
    DEFAULT_CONFIG = {
        "orientation": "portrait",
        "paper_size": "a4",
        "color_mode": "color",
        "copies": 1,
        "duplex": False,
        "pages": "all",
        "pages_per_sheet": 1,
        "scale": 100,
    }
    
    def __init__(self):
        """Initialize printer service."""
        self._config = self.DEFAULT_CONFIG.copy()
    
    def get_printer_queues(self) -> List[Dict[str, Any]]:
        """
        Get all printer queues with job information.
        
        Returns:
            List of printer info dictionaries
        """
        if sys.platform.startswith("win"):
            return self._get_windows_queues()
        else:
            return self._get_cups_queues()
    
    def _get_windows_queues(self) -> List[Dict[str, Any]]:
        """Get printer queues on Windows using PowerShell."""
        script = """
$printers = Get-Printer | Sort-Object -Property Name
$result = @()
foreach ($printer in $printers) {
    $jobs = @()
    
    # Method 1: Try Get-PrintJob (standard API)
    try {
        $jobs = Get-PrintJob -PrinterName $printer.Name -ErrorAction Stop | ForEach-Object {
            [PSCustomObject]@{
                id = $_.Id
                document = $_.DocumentName
                owner = $_.UserName
                status = ($_.JobStatus -join ',')
                submitted = $_.TimeSubmitted.ToString('o')
                totalPages = $_.TotalPages
                pagesPrinted = $_.PagesPrinted
                sizeBytes = $_.Size
            }
        }
    } catch {
        # Method 2: Fallback to WMI if Get-PrintJob fails
        try {
            $wmiJobs = Get-WmiObject -Class Win32_PrintJob -Filter "Name like '%$($printer.Name)%'" -ErrorAction Stop
            $jobs = $wmiJobs | ForEach-Object {
                [PSCustomObject]@{
                    id = $_.JobId
                    document = $_.Document
                    owner = $_.Owner
                    status = $_.Status
                    submitted = $_.TimeSubmitted
                    totalPages = $_.TotalPages
                    pagesPrinted = $_.PagesPrinted
                    sizeBytes = $_.Size
                }
            }
        } catch {
            # Method 3: Last resort - check spool directory for .SPL/.SHD files
            try {
                $spoolPath = Join-Path $env:windir 'System32\spool\PRINTERS'
                if (Test-Path $spoolPath) {
                    $spoolFiles = Get-ChildItem -Path $spoolPath -Filter '*.SPL' -ErrorAction SilentlyContinue
                    $jobs = $spoolFiles | ForEach-Object {
                        $shdFile = Join-Path $spoolPath ($_.BaseName + '.SHD')
                        [PSCustomObject]@{
                            id = $_.BaseName
                            document = $_.Name
                            owner = 'unknown'
                            status = 'spooling'
                            submitted = $_.CreationTime.ToString('o')
                            totalPages = $null
                            pagesPrinted = $null
                            sizeBytes = $_.Length
                        }
                    }
                }
            } catch {}
        }
    }
    
    $result += [PSCustomObject]@{
        name = $printer.Name
        status = $printer.PrinterStatus
        isDefault = $printer.Default
        jobs = @($jobs)
    }
}
$result | ConvertTo-Json -Depth 6
""".strip()
        
        try:
            result = subprocess.run(
                ["powershell", "-NoProfile", "-Command", script],
                capture_output=True,
                text=True,
                timeout=40,
            )
            
            if result.returncode != 0:
                raise RuntimeError(result.stderr.strip() or "Failed to query printers")
            
            output = result.stdout.strip() or "[]"
            raw_printers = json.loads(output)
            
            # Normalize results
            printers = []
            for printer in raw_printers:
                jobs = printer.get("jobs", []) or []
                printers.append({
                    "name": printer.get("name"),
                    "status": self._normalize_status(printer.get("status")),
                    "isDefault": printer.get("isDefault", False),
                    "jobs": [
                        {
                            "id": job.get("id"),
                            "document": job.get("document"),
                            "owner": job.get("owner"),
                            "status": job.get("status") or "in queue",
                            "submitted": job.get("submitted"),
                            "pagesPrinted": job.get("pagesPrinted"),
                            "totalPages": job.get("totalPages"),
                            "sizeBytes": job.get("sizeBytes"),
                        }
                        for job in jobs
                    ],
                })
            
            return printers
        
        except subprocess.TimeoutExpired:
            logger.error("Printer query timed out")
            return []
        except Exception as e:
            logger.error(f"Failed to get Windows printer queues: {e}")
            return []
    
    def _get_cups_queues(self) -> List[Dict[str, Any]]:
        """Get printer queues on Linux/macOS using CUPS."""
        printers = {}
        
        try:
            # Get printers
            printers_result = subprocess.run(
                ["lpstat", "-p"],
                capture_output=True,
                text=True,
                timeout=15
            )
            
            if printers_result.returncode == 0:
                for line in printers_result.stdout.splitlines():
                    parts = line.split()
                    if len(parts) >= 3 and parts[0] == "printer":
                        name = parts[1]
                        status_fragment = " ".join(parts[3:]).strip()
                        status = status_fragment.split(".")[0] if status_fragment else "unknown"
                        printers[name] = {
                            "name": name,
                            "status": status,
                            "isDefault": False,
                            "jobs": [],
                        }
            
            # Get jobs
            jobs_result = subprocess.run(
                ["lpstat", "-W", "not-completed", "-o"],
                capture_output=True,
                text=True,
                timeout=15,
            )
            
            if jobs_result.returncode == 0:
                for line in jobs_result.stdout.splitlines():
                    parts = line.split()
                    if not parts:
                        continue
                    job_id = parts[0]
                    printer_name = job_id.split("-", 1)[0]
                    owner = parts[1] if len(parts) > 1 else "unknown"
                    submitted = " ".join(parts[2:]) if len(parts) > 2 else ""
                    
                    job = {
                        "id": job_id,
                        "document": job_id,
                        "owner": owner,
                        "status": "pending",
                        "submitted": submitted,
                        "totalPages": None,
                        "pagesPrinted": None,
                        "sizeBytes": None,
                    }
                    
                    if printer_name in printers:
                        printers[printer_name]["jobs"].append(job)
                    else:
                        printers[printer_name] = {
                            "name": printer_name,
                            "status": "unknown",
                            "isDefault": False,
                            "jobs": [job],
                        }
            
            return list(printers.values())
        
        except Exception as e:
            logger.error(f"Failed to get CUPS printer queues: {e}")
            return []
    
    def _normalize_status(self, code: int) -> str:
        """Convert Windows printer status code to string."""
        status_map = {
            0: "ready",
            1: "paused",
            2: "error",
            3: "pending deletion",
            4: "paper jam",
            5: "paper out",
            6: "manual feed",
            7: "paper problem",
            8: "offline",
            9: "io active",
            10: "busy",
            11: "printing",
            12: "output bin full",
            13: "not available",
            14: "waiting",
            15: "processing",
            16: "initializing",
            17: "warming up",
            18: "toner low",
            19: "no toner",
            20: "page punt",
            21: "user intervention",
            22: "out of memory",
            23: "door open",
            24: "server unknown",
            25: "power save",
        }
        return status_map.get(int(code) if code is not None else -1, "unknown")
    
    def cancel_job(self, printer_name: str, job_id: str) -> bool:
        """
        Cancel a print job.
        
        Args:
            printer_name: Name of the printer
            job_id: ID of the job to cancel
            
        Returns:
            True if successful
        """
        if not printer_name or not job_id:
            raise ValueError("Printer name and job ID required")
        
        if sys.platform.startswith("win"):
            script = f'Remove-PrintJob -PrinterName "{printer_name}" -ID {job_id} -ErrorAction Stop'
            result = subprocess.run(
                ["powershell", "-NoProfile", "-Command", script],
                capture_output=True,
                text=True,
                timeout=20,
            )
            if result.returncode != 0:
                raise RuntimeError(result.stderr.strip() or "Failed to cancel job")
        else:
            job_identifier = job_id if "-" in job_id else f"{printer_name}-{job_id}"
            result = subprocess.run(
                ["cancel", job_identifier],
                capture_output=True,
                text=True,
                timeout=15
            )
            if result.returncode != 0:
                raise RuntimeError(result.stderr.strip() or "Failed to cancel job")
        
        return True
    
    def clear_queue(self, printer_name: str) -> int:
        """
        Clear all jobs from a printer queue.
        
        Args:
            printer_name: Name of the printer
            
        Returns:
            Number of jobs cleared
        """
        queues = self.get_printer_queues()
        
        # Find the printer
        printer = None
        for q in queues:
            if q["name"] == printer_name:
                printer = q
                break
        
        if not printer:
            raise ValueError(f"Printer not found: {printer_name}")
        
        cleared = 0
        for job in printer.get("jobs", []):
            try:
                self.cancel_job(printer_name, str(job["id"]))
                cleared += 1
            except Exception as e:
                logger.warning(f"Failed to cancel job {job['id']}: {e}")
        
        return cleared
    
    def get_default_printer(self) -> Optional[str]:
        """Get the default printer name."""
        if sys.platform.startswith("win"):
            try:
                import win32print
                return win32print.GetDefaultPrinter()
            except ImportError:
                # Fallback to PowerShell
                result = subprocess.run(
                    ["powershell", "-NoProfile", "-Command",
                     "(Get-Printer | Where-Object {$_.Default -eq $true}).Name"],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                if result.returncode == 0:
                    return result.stdout.strip()
        else:
            result = subprocess.run(
                ["lpstat", "-d"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                # Parse "system default destination: PRINTER_NAME"
                parts = result.stdout.strip().split(":")
                if len(parts) >= 2:
                    return parts[1].strip()
        
        return None
    
    def get_diagnostics(self) -> Dict[str, Any]:
        """Get printer diagnostics information."""
        diagnostics = {
            "platform": sys.platform,
            "default_printer": self.get_default_printer(),
            "printers": [],
            "spooler_status": "unknown",
        }
        
        try:
            queues = self.get_printer_queues()
            diagnostics["printers"] = [
                {
                    "name": p["name"],
                    "status": p["status"],
                    "isDefault": p["isDefault"],
                    "job_count": len(p.get("jobs", []))
                }
                for p in queues
            ]
            diagnostics["printer_count"] = len(queues)
        except Exception as e:
            diagnostics["error"] = str(e)
        
        # Check spooler status (Windows)
        if sys.platform.startswith("win"):
            try:
                result = subprocess.run(
                    ["powershell", "-NoProfile", "-Command",
                     "(Get-Service -Name Spooler).Status"],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                if result.returncode == 0:
                    diagnostics["spooler_status"] = result.stdout.strip().lower()
            except:
                pass
        
        return diagnostics
    
    def run_diagnostic(self, test_type: str) -> Dict[str, Any]:
        """Run a diagnostic test."""
        if test_type == "status":
            return self.get_diagnostics()
        elif test_type == "connectivity":
            return {
                "default_printer": self.get_default_printer(),
                "can_query_printers": len(self.get_printer_queues()) > 0
            }
        else:
            return {"error": f"Unknown test type: {test_type}"}
    
    def get_print_config(self) -> Dict[str, Any]:
        """Get current print configuration."""
        return self._config.copy()
    
    def update_print_config(self, updates: Dict[str, Any]) -> Dict[str, Any]:
        """Update print configuration."""
        valid_keys = set(self.DEFAULT_CONFIG.keys())
        
        for key, value in updates.items():
            if key in valid_keys:
                self._config[key] = value
        
        return self._config.copy()
    
    def list_printers(self) -> List[Dict[str, Any]]:
        """
        List all available printers.
        
        Returns:
            List of printer info dictionaries
        """
        queues = self.get_printer_queues()
        return [
            {
                "name": p["name"],
                "status": p["status"],
                "isDefault": p.get("isDefault", False),
            }
            for p in queues
        ]
    
    def get_printer_status(self, printer_name: str) -> Dict[str, Any]:
        """
        Get status of a specific printer.
        
        Args:
            printer_name: Name of the printer
            
        Returns:
            Status dictionary
        """
        queues = self.get_printer_queues()
        
        for printer in queues:
            if printer["name"] == printer_name:
                return {
                    "online": printer["status"] not in ["offline", "error"],
                    "state": printer["status"],
                    "jobs_pending": len(printer.get("jobs", []))
                }
        
        return {
            "online": False,
            "state": "not found",
            "jobs_pending": 0
        }
    
    def get_printer_capabilities(self, printer_name: str) -> Dict[str, Any]:
        """
        Get capabilities of a specific printer.
        
        Args:
            printer_name: Name of the printer
            
        Returns:
            Capabilities dictionary
        """
        # Default capabilities - in a real implementation, query the printer
        return {
            "color": True,
            "duplex": False,
            "paper_sizes": ["A4", "Letter", "Legal", "A3", "A5"],
            "default_paper_size": "A4",
            "resolutions": [300, 600, 1200],
            "default_resolution": 300
        }
    
    def set_default_printer(self, printer_name: str) -> bool:
        """
        Set the default printer.
        
        Args:
            printer_name: Name of printer to set as default
            
        Returns:
            True if successful
        """
        if sys.platform.startswith("win"):
            try:
                script = f'(Get-WmiObject -Query "SELECT * FROM Win32_Printer WHERE Name=\'{printer_name}\'").SetDefaultPrinter()'
                result = subprocess.run(
                    ["powershell", "-NoProfile", "-Command", script],
                    capture_output=True,
                    text=True,
                    timeout=15
                )
                return result.returncode == 0
            except Exception as e:
                logger.error(f"Failed to set default printer: {e}")
                return False
        else:
            try:
                result = subprocess.run(
                    ["lpoptions", "-d", printer_name],
                    capture_output=True,
                    text=True,
                    timeout=15
                )
                return result.returncode == 0
            except Exception as e:
                logger.error(f"Failed to set default printer: {e}")
                return False
    
    def get_settings(self) -> Dict[str, Any]:
        """
        Get global print settings.
        
        Returns:
            Settings dictionary
        """
        return {
            "default_copies": self._config.get("copies", 1),
            "default_color": self._config.get("color_mode", "color") == "color",
            "default_duplex": self._config.get("duplex", False),
            "default_paper_size": self._config.get("paper_size", "a4").upper(),
            "auto_fit_page": True
        }
    
    def update_settings(self, settings: Dict[str, Any]) -> None:
        """
        Update global print settings.
        
        Args:
            settings: Settings dictionary
        """
        if "default_copies" in settings:
            self._config["copies"] = settings["default_copies"]
        if "default_color" in settings:
            self._config["color_mode"] = "color" if settings["default_color"] else "grayscale"
        if "default_duplex" in settings:
            self._config["duplex"] = settings["default_duplex"]
        if "default_paper_size" in settings:
            self._config["paper_size"] = settings["default_paper_size"].lower()
    
    def get_queue(self, printer_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get print queue for a printer.
        
        Args:
            printer_name: Name of printer, or None for default
            
        Returns:
            List of jobs in queue
        """
        queues = self.get_printer_queues()
        
        if printer_name:
            for printer in queues:
                if printer["name"] == printer_name:
                    return printer.get("jobs", [])
            return []
        else:
            # Return all jobs from all printers
            all_jobs = []
            for printer in queues:
                for job in printer.get("jobs", []):
                    job["printer"] = printer["name"]
                    all_jobs.append(job)
            return all_jobs
    
    def pause_queue(self, printer_name: Optional[str] = None) -> bool:
        """
        Pause print queue.
        
        Args:
            printer_name: Name of printer, or None for default
            
        Returns:
            True if successful
        """
        target = printer_name or self.get_default_printer()
        if not target:
            return False
        
        if sys.platform.startswith("win"):
            try:
                script = f'Set-Printer -Name "{target}" -PrinterStatus Paused'
                result = subprocess.run(
                    ["powershell", "-NoProfile", "-Command", script],
                    capture_output=True,
                    timeout=15
                )
                return result.returncode == 0
            except Exception:
                return False
        else:
            try:
                result = subprocess.run(
                    ["cupsdisable", target],
                    capture_output=True,
                    timeout=15
                )
                return result.returncode == 0
            except Exception:
                return False
    
    def resume_queue(self, printer_name: Optional[str] = None) -> bool:
        """
        Resume print queue.
        
        Args:
            printer_name: Name of printer, or None for default
            
        Returns:
            True if successful
        """
        target = printer_name or self.get_default_printer()
        if not target:
            return False
        
        if sys.platform.startswith("win"):
            try:
                script = f'Set-Printer -Name "{target}" -PrinterStatus Normal'
                result = subprocess.run(
                    ["powershell", "-NoProfile", "-Command", script],
                    capture_output=True,
                    timeout=15
                )
                return result.returncode == 0
            except Exception:
                return False
        else:
            try:
                result = subprocess.run(
                    ["cupsenable", target],
                    capture_output=True,
                    timeout=15
                )
                return result.returncode == 0
            except Exception:
                return False
    
    def reorder_queue(self, printer_name: Optional[str], order: List[str]) -> bool:
        """
        Reorder print queue (not typically supported by OS print systems).
        
        Args:
            printer_name: Name of printer
            order: New order of job IDs
            
        Returns:
            True if successful (always False for most systems)
        """
        # Most print systems don't support reordering
        logger.warning("Queue reordering is not supported by most print systems")
        return False
    
    def get_queue_stats(self) -> Dict[str, Any]:
        """
        Get print queue statistics.
        
        Returns:
            Statistics dictionary
        """
        queues = self.get_printer_queues()
        
        stats = {
            "pending": 0,
            "printing": 0,
            "completed": 0,
            "failed": 0,
            "total": 0
        }
        
        for printer in queues:
            for job in printer.get("jobs", []):
                stats["total"] += 1
                status = (job.get("status") or "").lower()
                if "print" in status:
                    stats["printing"] += 1
                elif "error" in status or "fail" in status:
                    stats["failed"] += 1
                else:
                    stats["pending"] += 1
        
        return stats
