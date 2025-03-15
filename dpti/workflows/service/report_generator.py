from typing import Protocol

class ReportGenerator(Protocol):
    def generate_figures(self, data, figure_specs): ...
    def create_markdown_report(self, title, sections, figures=None): ...
    def publish_results(self, destination, files): ...
    def archive_artifacts(self, run_id, artifacts): ...




class PrefectReportGenerator(object): # implement ReportGenerator
    def __init__(self, output_dir=None, template_dir=None):
        self.output_dir = output_dir
        self.template_dir = template_dir
        
    def generate_figures(self, data, figure_specs):
        """Generate figures based on data"""
        # Implement...
        
    def create_markdown_report(self, title, sections, figures=None):
        """Create Markdown report"""
        # Implement...
        
    def publish_results(self, destination, files):
        """Publish results to specified destination"""
        # Implement...
        
    def archive_artifacts(self, run_id, artifacts):
        """Archive run results"""
        # Implement...
