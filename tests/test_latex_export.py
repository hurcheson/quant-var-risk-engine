import unittest
from models.latex_report_generator import (
    LaTeXReportGenerator,
    get_gbm_latex,
    get_var_latex,
    get_cvar_latex
)

class TestLaTeXReportGenerator(unittest.TestCase):
    def setUp(self):
        self.sim_results = {
            "var_amount": 45000.50,
            "cvar_amount": 58000.75,
            "confidence_level": 0.95,
            "time_horizon_days": 1,
            "simulations": 10000,
            "mean_return": 0.0007,
            "volatility": 0.015
        }
        self.portfolio_value = 1_000_000.0

    def test_latex_equation_helpers(self):
        gbm_tex = get_gbm_latex()
        var_tex = get_var_latex()
        cvar_tex = get_cvar_latex()

        self.assertIn(r"\mu S_t dt", gbm_tex)
        self.assertIn(r"\text{VaR}", var_tex)
        self.assertIn(r"\text{CVaR}", cvar_tex)

    def test_generate_tex_document_structure(self):
        generator = LaTeXReportGenerator(self.sim_results, self.portfolio_value)
        tex_doc = generator.generate_tex_document()

        self.assertIn(r"\documentclass{article}", tex_doc)
        self.assertIn(r"\begin{document}", tex_doc)
        self.assertIn(r"\end{document}", tex_doc)
        self.assertIn(r"Executive Summary", tex_doc)
        self.assertIn(r"Mathematical Methodology", tex_doc)
        self.assertIn("$1,000,000.00", tex_doc)
        self.assertIn("$45,000.50", tex_doc)
        self.assertIn("$58,000.75", tex_doc)
        self.assertIn(r"95\%", tex_doc)

if __name__ == "__main__":
    unittest.main()
