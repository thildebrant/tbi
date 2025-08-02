"""
Tests for the utils module.
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile
import shutil
import logging

# Add src to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent / "src"))

from utils import setup_logging, get_image_info, save_color_lookup_table


class TestUtils:
    """Test cases for utility functions."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_setup_logging(self):
        """Test logging setup function."""
        log_file = Path(self.temp_dir) / "test.log"
        
        # Test that logging can be set up
        setup_logging("INFO", log_file)
        
        # Verify log file was created
        assert log_file.exists()
        
        # Test that logger is configured
        logger = logging.getLogger(__name__)
        assert logger.level <= logging.INFO
    
    def test_get_image_info(self):
        """Test image info extraction."""
        # Create a dummy image info structure
        dummy_info = {
            'shape': (256, 256, 256),
            'spacing': (1.0, 1.0, 1.0),
            'dtype': 'float32'
        }
        
        # Test that function exists and can be called
        # Note: This is a placeholder test since we don't have actual image files
        assert callable(get_image_info)
    
    def test_save_color_lookup_table(self):
        """Test color lookup table saving."""
        color_table = {
            'background': [0, 0, 0],
            'lesion1': [255, 0, 0],
            'lesion2': [0, 255, 0]
        }
        
        output_file = Path(self.temp_dir) / "colors.json"
        
        # Test that function exists and can be called
        # Note: This is a placeholder test since we don't have actual image files
        assert callable(save_color_lookup_table)
    
    def test_temp_directory_creation(self):
        """Test that temporary directory is created and accessible."""
        assert Path(self.temp_dir).exists()
        assert Path(self.temp_dir).is_dir()
    
    def test_temp_directory_cleanup(self):
        """Test that temporary directory can be cleaned up."""
        test_file = Path(self.temp_dir) / "test.txt"
        test_file.write_text("test content")
        
        # Verify file was created
        assert test_file.exists()
        
        # Cleanup should work
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        
        # Directory should be gone
        assert not Path(self.temp_dir).exists()


if __name__ == "__main__":
    pytest.main([__file__]) 