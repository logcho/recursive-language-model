import unittest
from unittest.mock import patch, MagicMock, mock_open
import os
import sys

# Add root directory to python path to import main
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import main

class TestMainPDFParsing(unittest.TestCase):
    @patch("os.path.exists")
    @patch("main.get_model")
    @patch("main.RLMEngine")
    def test_pdf_parsing_success(self, mock_engine, mock_get_model, mock_exists):
        """Verify that when a .pdf file is passed, PdfReader is used to extract text."""
        mock_exists.return_value = True
        
        # Mock sys.argv
        test_args = [
            "main.py",
            "--provider", "mock",
            "--query", "Summarize this PDF",
            "--context-file", "dummy.pdf"
        ]
        
        # Mock PdfReader and pages
        mock_reader = MagicMock()
        mock_page1 = MagicMock()
        mock_page1.extract_text.return_value = "This is page 1 content."
        mock_page2 = MagicMock()
        mock_page2.extract_text.return_value = "This is page 2 content."
        mock_reader.pages = [mock_page1, mock_page2]
        
        with patch("sys.argv", test_args), \
             patch("pypdf.PdfReader", return_value=mock_reader) as mock_pdf_reader:
            
            main.main()
            
            # Verify PdfReader was called with the correct path
            mock_pdf_reader.assert_called_once_with("dummy.pdf")
            # Verify the engine was run with the concatenated text
            mock_engine.return_value.run.assert_called_once_with(
                "Summarize this PDF",
                "This is page 1 content.\nThis is page 2 content."
            )

    @patch("os.path.exists")
    @patch("main.get_model")
    @patch("main.RLMEngine")
    def test_standard_txt_parsing(self, mock_engine, mock_get_model, mock_exists):
        """Verify that when a non-PDF file is passed, it is read as a regular text file."""
        mock_exists.return_value = True
        
        test_args = [
            "main.py",
            "--provider", "mock",
            "--query", "Summarize this text file",
            "--context-file", "dummy.txt"
        ]
        
        file_content = "This is standard text content."
        
        with patch("sys.argv", test_args), \
             patch("builtins.open", mock_open(read_data=file_content)) as mock_file:
            
            main.main()
            
            # Verify standard file open was called
            mock_file.assert_any_call("dummy.txt", "r", encoding="utf-8")
            # Verify engine run was called with text file contents
            mock_engine.return_value.run.assert_called_once_with(
                "Summarize this text file",
                file_content
            )

if __name__ == "__main__":
    unittest.main()
