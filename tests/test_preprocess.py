"""
Tests for the preprocessing module.
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile
import shutil

# Add src to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent / "src"))

from preprocess import MRIPreprocessor


class TestMRIPreprocessor:
    """Test cases for MRIPreprocessor class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.preprocessor = MRIPreprocessor()
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_preprocessor_initialization(self):
        """Test that preprocessor initializes correctly."""
        assert self.preprocessor is not None
        assert hasattr(self.preprocessor, 'target_spacing')
        assert hasattr(self.preprocessor, 'target_shape')
    
    def test_config_loading(self):
        """Test that configuration loads correctly."""
        # This would test loading from config.yaml
        # For now, just test that the method exists
        assert hasattr(self.preprocessor, 'load_config')
    
    def test_bias_correction_parameters(self):
        """Test bias correction parameter validation."""
        # Test that bias correction parameters are valid
        assert isinstance(self.preprocessor.bias_correction_iterations, list)
        assert len(self.preprocessor.bias_correction_iterations) > 0
        assert all(isinstance(x, int) and x > 0 for x in self.preprocessor.bias_correction_iterations)
    
    def test_intensity_normalization_parameters(self):
        """Test intensity normalization parameter validation."""
        # Test that normalization parameters are valid
        assert 0 <= self.preprocessor.percentile_low <= 100
        assert 0 <= self.preprocessor.percentile_high <= 100
        assert self.preprocessor.percentile_low < self.preprocessor.percentile_high


if __name__ == "__main__":
    pytest.main([__file__]) 