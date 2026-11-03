"""
PrintChakra Backend - Printer Service

Service for printer management, queue operations, and diagnostics.
"""

import os
import sys
import subprocess
import logging
from typing import List, Dict, Any, Optional

from app.core.paths import IS_CLOUD

from .queue_discovery_mixin import PrinterQueueDiscoveryMixin

logger = logging.getLogger(__name__)


class PrinterService(PrinterQueueDiscoveryMixin):
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
        if IS_CLOUD:
            return "Browser Print"
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
        if IS_CLOUD:
            return [{"name": "Browser Print", "status": "Ready", "isDefault": True}]
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
