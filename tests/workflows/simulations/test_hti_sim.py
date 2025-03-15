import os
import tempfile
from unittest.mock import Mock, patch
from pathlib import Path
from typing import List

from dpti.workflows.simulations.base import SimulationBase
from dpti.workflows.service.file_handler import IOHandler

from dpti.workflows.simulations.hti_sim import HTISimulationSettings, HTISimulation, HTIIntegrationPathTemplate, HTIIntegrationPath
from dpti.workflows.simulations.hti_sim import SolidHTI
import pytest
from unittest.mock import Mock, patch, mock_open
from pathlib import Path
from typing import List

class TestHTISimulation:
    def setup_method(self):
        """Setup test fixtures"""
        # Create a mock context manager for subdir_context
        self.mock_context = Mock()
        self.mock_context.__enter__ = Mock(return_value=self.mock_context)
        self.mock_context.__exit__ = Mock(return_value=None)
        
        # Create mock io_handler with context manager support
        self.mock_io_handler = Mock(spec=IOHandler)
        self.mock_io_handler.flow_running_dir = "/mock/flow/dir"
        self.mock_io_handler.flow_trigger_dir = "/mock/trigger/dir"
        self.mock_io_handler.subdir_context.return_value = self.mock_context
        
        # Initialize settings
        self.node_settings = HTISimulationSettings(
            equi_conf="test.conf",
            model="test.pb",
            lambda_lj_on=["0:1:0.2"],
            lambda_deep_on=["0:1:0.1"],
            lambda_spring_off=["0:0.5:0.1", "0.5:1:0.1", "1"],
            protect_eps=1e-6,
            mass_map=[118.71],
            spring_k=0.02,
            soft_param=dict(sigma_0_0=2.7,
                            epsilon=0.030,
                            activation=0.5, n=1.0,
                            alpha_lj=0.5,
                            rcut=6.0),
            crystal="frenkel",
            langevin=True,
            nsteps=10000,
            timestep=0.002,
            thermo_freq=10,
            stat_skip=1000,
            stat_bsize=100,
            temp=300,
            pres=10000,
            switch="three-step",  # Set initial switch mode
            if_liquid=False,
            if_water=False,
            if_meam=False,
            meam_model=None,
        )

    def test_integration_path_template(self):
        """Test HTIIntegrationPathTemplate functionality"""
        tmpl = HTIIntegrationPathTemplate(
            field_name='lambda_deep_on',
            step_name='deep_on',
            subtasks_dirname='01.deep_on'
        )
        
        # Create path instance from template
        path = tmpl(seq_list=['0:1:0.1'])
        
        # Verify path properties
        assert path.template.field_name == 'lambda_deep_on'
        assert path.template.step_name == 'deep_on'
        assert path.template == tmpl
        assert len(path.all_lambda) > 0
        assert isinstance(path.all_lambda, list)

    def test_prepare_integration_paths(self):
        """Test integration path generation with different switch modes"""
        # Create SolidHTI instance
        hti_adapter = SolidHTI(node_settings=self.node_settings)
        hti_adapter.io_handler = self.mock_io_handler
        
        # Test one-step mode
        paths = hti_adapter.generate_integration_path_list(
            switch=self.node_settings.switch
        )
        assert len(paths) == 3
        assert paths[0].template.step_name == "lj_on"
        
        # Test three-step mode
        self.node_settings.switch = "three-step"
        paths = hti_adapter.generate_integration_path_list(
            switch=self.node_settings.switch
        )
        assert len(paths) == 3
        assert [p.template.step_name for p in paths] == ["lj_on", "deep_on", "spring_off"]

    @patch('builtins.open', new_callable=mock_open)
    def test_prepare_file_handling(self, mock_file):
        """Test file handling during preparation phase"""
        # Create SolidHTI instance
        # hti_simulation = SolidHTI(node_settings=self.node_settings)
        hti_simulation = HTISimulation(node_settings=self.node_settings)
        hti_simulation.io_handler = self.mock_io_handler
        hti_simulation.node_upstream_data={'conf_file':'nvt_out.lmp'}
        
        # Call prepare method
        hti_simulation._prepare()
        
        # Verify context manager was called
        self.mock_io_handler.subdir_context.assert_called()
        self.mock_context.__enter__.assert_called()
        self.mock_context.__exit__.assert_called()
        
        # Verify file operations
        self.mock_io_handler.upload_file.assert_called()
        self.mock_io_handler.write_pure_file.assert_called()
        
        # Verify in.json content
        calls = self.mock_io_handler.write_pure_file.call_args_list
        in_json_call = [c for c in calls if c[1]['file_path'] == 'in.json'][0]
        in_json_content = in_json_call[1]['file_content']
        assert 'equi_conf' in in_json_content
        assert 'temp' in in_json_content
