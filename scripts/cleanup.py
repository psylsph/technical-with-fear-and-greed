#!/usr/bin/env python3
"""
Cleanup script for removing unused files and consolidating duplicate code.

This script helps clean up the codebase by:
1. Removing duplicate trading engine files
2. Archiving unused trading modules
3. Cleaning up temporary test/debug files
4. Consolidating similar functionality

Usage:
    python scripts/cleanup.py --dry-run    # Show what would be removed
    python scripts/cleanup.py              # Actually remove files
"""

import shutil
import argparse
from pathlib import Path
from typing import List, Tuple


class CodeCleanup:
    """Handle cleanup of unused files and duplicate code."""
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root).resolve()
        self.archive_dir = self.project_root / "archive"
        
    def identify_files_to_remove(self) -> List[Tuple[Path, str]]:
        """Identify files that should be removed or archived.
        
        Returns:
            List of (file_path, reason) tuples
        """
        files_to_remove = []
        
        # 1. Duplicate trading engine files
        trading_engine_files = [
            self.project_root / "src" / "trading" / "trading_engine_broken.py",
            self.project_root / "src" / "trading" / "trading_engine_fixed.py",
        ]
        
        for file_path in trading_engine_files:
            if file_path.exists():
                files_to_remove.append((file_path, "Duplicate trading engine file"))
        
        # 2. Temporary test/debug files in root directory
        temp_files = [
            self.project_root / "debug_live.py",
            self.project_root / "test_live_reach.py",
            self.project_root / "test_live_simple.py",
            self.project_root / "test_minimal.py",
            self.project_root / "test_monitor.py",
            self.project_root / "test_env.py",
            self.project_root / "test_telegram_bot.py",
        ]
        
        for file_path in temp_files:
            if file_path.exists():
                files_to_remove.append((file_path, "Temporary test/debug file"))
        
        # 3. Large test file that should be in tests/ directory
        large_test = self.project_root / "test_multi_asset.py"
        if large_test.exists():
            files_to_remove.append((large_test, "Large test file - should be in tests/ directory"))
        
        return files_to_remove
    
    def identify_modules_to_archive(self) -> List[Tuple[Path, str]]:
        """Identify potentially unused modules that should be archived.
        
        Returns:
            List of (module_path, reason) tuples
        """
        modules_to_archive = []
        
        # Trading modules that appear to be unused
        trading_modules = [
            "anomaly_detection.py",
            "onchain_metrics.py",
            "monte_carlo_simulation.py",
            "walk_forward_analysis.py",
            "health_checks.py",
            "config_manager.py",
            "filters.py",
            "execution_features.py",
            "data_quality_framework.py",
            "property_based_testing.py",
            "automatic_recovery.py",
            "intelligent_caching.py",
            "api_circuit_breaker.py",
            "market_regime.py",
            "rate_limiter.py",
            "correlation_analysis.py",
            "data_lineage.py",
            "signal_confluence.py",
            "data_validation.py",
            "email_notifications.py",
            "trade_analytics.py",
            "risk_controls.py",
            "state_manager.py",
            "event_bus.py",
            "change_data_capture.py",
            "stress_testing.py",
            "out_of_sample_testing.py",
            "advanced_orders.py",
        ]
        
        trading_dir = self.project_root / "src" / "trading"
        for module in trading_modules:
            module_path = trading_dir / module
            if module_path.exists():
                modules_to_archive.append((module_path, "Potentially unused trading module"))
        
        # ML module that appears to be unused
        ml_module = self.project_root / "src" / "ml" / "advanced_ml_models.py"
        if ml_module.exists():
            modules_to_archive.append((ml_module, "Potentially unused ML module"))
        
        return modules_to_archive
    
    def create_archive_structure(self):
        """Create archive directory structure."""
        archive_dirs = [
            self.archive_dir,
            self.archive_dir / "trading",
            self.archive_dir / "ml",
        ]
        
        for dir_path in archive_dirs:
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"Created directory: {dir_path.relative_to(self.project_root)}")
    
    def archive_module(self, module_path: Path, dry_run: bool = True) -> bool:
        """Archive a module by moving it to the archive directory.
        
        Args:
            module_path: Path to the module to archive
            dry_run: If True, only show what would be done
            
        Returns:
            True if successful or would be successful in dry run
        """
        if not module_path.exists():
            print(f"⚠️  Module does not exist: {module_path}")
            return False
        
        # Determine archive subdirectory based on module location
        if "trading" in str(module_path):
            archive_subdir = self.archive_dir / "trading"
        elif "ml" in str(module_path):
            archive_subdir = self.archive_dir / "ml"
        else:
            archive_subdir = self.archive_dir
        
        archive_path = archive_subdir / module_path.name
        
        if dry_run:
            print(f"📦 [DRY RUN] Would archive: {module_path.relative_to(self.project_root)}")
            print(f"            -> {archive_path.relative_to(self.project_root)}")
            return True
        
        try:
            shutil.move(str(module_path), str(archive_path))
            print(f"📦 Archived: {module_path.relative_to(self.project_root)}")
            print(f"            -> {archive_path.relative_to(self.project_root)}")
            return True
        except Exception as e:
            print(f"❌ Failed to archive {module_path}: {e}")
            return False
    
    def remove_file(self, file_path: Path, dry_run: bool = True) -> bool:
        """Remove a file.
        
        Args:
            file_path: Path to the file to remove
            dry_run: If True, only show what would be done
            
        Returns:
            True if successful or would be successful in dry run
        """
        if not file_path.exists():
            print(f"⚠️  File does not exist: {file_path}")
            return False
        
        if dry_run:
            print(f"🗑️  [DRY RUN] Would remove: {file_path.relative_to(self.project_root)}")
            return True
        
        try:
            file_path.unlink()
            print(f"🗑️  Removed: {file_path.relative_to(self.project_root)}")
            return True
        except Exception as e:
            print(f"❌ Failed to remove {file_path}: {e}")
            return False
    
    def run_cleanup(self, dry_run: bool = True):
        """Run the cleanup process.
        
        Args:
            dry_run: If True, only show what would be done without making changes
        """
        print("=" * 60)
        print("CODE CLEANUP UTILITY")
        print("=" * 60)
        
        if dry_run:
            print("🔍 DRY RUN MODE - No files will be modified")
        else:
            print("⚡ LIVE MODE - Files will be archived/removed")
        print()
        
        # Create archive directory if needed
        if not dry_run:
            self.create_archive_structure()
        
        # Identify and process files to remove
        files_to_remove = self.identify_files_to_remove()
        
        print("📋 FILES TO REMOVE:")
        print("-" * 40)
        if not files_to_remove:
            print("No files identified for removal")
        else:
            for file_path, reason in files_to_remove:
                print(f"• {file_path.relative_to(self.project_root)}")
                print(f"  Reason: {reason}")
            
            print()
            print("Processing file removal...")
            removed_count = 0
            for file_path, _ in files_to_remove:
                if self.remove_file(file_path, dry_run):
                    removed_count += 1
            
            print(f"\n✅ {removed_count}/{len(files_to_remove)} files processed")
        
        print()
        
        # Identify and process modules to archive
        modules_to_archive = self.identify_modules_to_archive()
        
        print("📦 MODULES TO ARCHIVE:")
        print("-" * 40)
        if not modules_to_archive:
            print("No modules identified for archiving")
        else:
            for module_path, reason in modules_to_archive:
                print(f"• {module_path.relative_to(self.project_root)}")
                print(f"  Reason: {reason}")
            
            print()
            print("Processing module archiving...")
            archived_count = 0
            for module_path, _ in modules_to_archive:
                if self.archive_module(module_path, dry_run):
                    archived_count += 1
            
            print(f"\n✅ {archived_count}/{len(modules_to_archive)} modules processed")
        
        print()
        print("=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"Files to remove: {len(files_to_remove)}")
        print(f"Modules to archive: {len(modules_to_archive)}")
        print(f"Total items: {len(files_to_remove) + len(modules_to_archive)}")
        
        if dry_run:
            print("\n💡 Run without --dry-run to actually perform cleanup")
        else:
            print("\n🎉 Cleanup completed!")
        
        # Show disk space savings estimate
        if files_to_remove:
            total_size = sum(f.stat().st_size for f, _ in files_to_remove if f.exists())
            print(f"\n💾 Estimated space savings: {total_size / 1024:.1f} KB")


def main():
    parser = argparse.ArgumentParser(description="Clean up unused files and duplicate code")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes"
    )
    parser.add_argument(
        "--project-root",
        default=".",
        help="Project root directory (default: current directory)"
    )
    
    args = parser.parse_args()
    
    cleanup = CodeCleanup(args.project_root)
    cleanup.run_cleanup(dry_run=args.dry_run)


if __name__ == "__main__":
    main()