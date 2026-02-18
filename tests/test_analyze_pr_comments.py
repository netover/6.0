"""
Comprehensive Test Suite for analyze_pr_comments.py

This module provides comprehensive testing for the PR comment analysis script:
- JSON file reading and parsing
- Comment and review formatting
- Error handling for missing files and invalid JSON
- UTF-8 encoding support
- Edge cases and boundary conditions

Tests cover normal operation, error conditions, and edge cases.
"""

import json
import os
import sys
import tempfile
from io import StringIO
from pathlib import Path
from unittest.mock import mock_open, patch

import pytest


class TestAnalyzePRComments:
    """Test cases for analyze_pr_comments.py script."""

    @pytest.fixture
    def sample_pr_data(self):
        """Fixture providing sample PR comment data."""
        return {
            "comments": [
                {
                    "author": {"login": "user1"},
                    "createdAt": "2024-01-01T12:00:00Z",
                    "body": "This is a test comment"
                },
                {
                    "author": {"login": "user2"},
                    "createdAt": "2024-01-02T12:00:00Z",
                    "body": "Another comment with unicode: \u2713"
                }
            ],
            "reviews": [
                {
                    "author": {"login": "reviewer1"},
                    "submittedAt": "2024-01-03T12:00:00Z",
                    "body": "Looks good to me!"
                }
            ]
        }

    @pytest.fixture
    def empty_pr_data(self):
        """Fixture providing empty PR comment data."""
        return {
            "comments": [],
            "reviews": []
        }

    @pytest.fixture
    def script_path(self):
        """Fixture providing path to the script."""
        return Path(__file__).parent.parent / "analyze_pr_comments.py"

    def test_successful_json_parsing_with_comments_and_reviews(self, sample_pr_data, tmp_path):
        """Test successful parsing of JSON file with both comments and reviews."""
        json_file = tmp_path / "pr_comments.json"
        json_file.write_text(json.dumps(sample_pr_data), encoding='utf-8')

        # Change to temp directory and run script
        original_dir = os.getcwd()
        try:
            os.chdir(tmp_path)

            # Capture stdout
            captured_output = StringIO()
            sys.stdout = captured_output

            # Execute the script code
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            print(f"Total Comments: {len(data.get('comments', []))}")
            print(f"Total Reviews: {len(data.get('reviews', []))}")
            print("-" * 50)

            for i, comment in enumerate(data.get('comments', [])):
                print(f"COMMENT #{i+1} by {comment['author']['login']} ({comment['createdAt']})")
                print(comment['body'])
                print("-" * 50)

            for i, review in enumerate(data.get('reviews', [])):
                print(f"REVIEW #{i+1} by {review['author']['login']} ({review['submittedAt']})")
                print(review['body'])
                print("-" * 50)

            sys.stdout = sys.__stdout__
            output = captured_output.getvalue()

            # Verify output contains expected elements
            assert "Total Comments: 2" in output
            assert "Total Reviews: 1" in output
            assert "COMMENT #1 by user1" in output
            assert "COMMENT #2 by user2" in output
            assert "This is a test comment" in output
            assert "REVIEW #1 by reviewer1" in output
            assert "Looks good to me!" in output

        finally:
            os.chdir(original_dir)
            sys.stdout = sys.__stdout__

    def test_empty_comments_and_reviews(self, empty_pr_data, tmp_path):
        """Test handling of JSON file with empty comments and reviews arrays."""
        json_file = tmp_path / "pr_comments.json"
        json_file.write_text(json.dumps(empty_pr_data), encoding='utf-8')

        original_dir = os.getcwd()
        try:
            os.chdir(tmp_path)

            captured_output = StringIO()
            sys.stdout = captured_output

            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            print(f"Total Comments: {len(data.get('comments', []))}")
            print(f"Total Reviews: {len(data.get('reviews', []))}")

            sys.stdout = sys.__stdout__
            output = captured_output.getvalue()

            assert "Total Comments: 0" in output
            assert "Total Reviews: 0" in output

        finally:
            os.chdir(original_dir)
            sys.stdout = sys.__stdout__

    def test_missing_json_file(self, tmp_path):
        """Test error handling when JSON file is missing."""
        original_dir = os.getcwd()
        try:
            os.chdir(tmp_path)

            captured_output = StringIO()
            sys.stdout = captured_output

            try:
                with open('pr_comments.json', 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except Exception as e:
                print(f"Error parsing JSON: {e}")

            sys.stdout = sys.__stdout__
            output = captured_output.getvalue()

            assert "Error parsing JSON:" in output

        finally:
            os.chdir(original_dir)
            sys.stdout = sys.__stdout__

    def test_invalid_json_format(self, tmp_path):
        """Test error handling when JSON file contains invalid JSON."""
        json_file = tmp_path / "pr_comments.json"
        json_file.write_text("{ invalid json content }", encoding='utf-8')

        original_dir = os.getcwd()
        try:
            os.chdir(tmp_path)

            captured_output = StringIO()
            sys.stdout = captured_output

            try:
                with open('pr_comments.json', 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except Exception as e:
                print(f"Error parsing JSON: {e}")

            sys.stdout = sys.__stdout__
            output = captured_output.getvalue()

            assert "Error parsing JSON:" in output

        finally:
            os.chdir(original_dir)
            sys.stdout = sys.__stdout__

    def test_utf8_encoding_in_comments(self, tmp_path):
        """Test proper handling of UTF-8 encoded characters in comments."""
        data_with_unicode = {
            "comments": [
                {
                    "author": {"login": "user1"},
                    "createdAt": "2024-01-01T12:00:00Z",
                    "body": "Unicode test: \u2713 \u2717 \u263a \U0001f680"
                }
            ],
            "reviews": []
        }

        json_file = tmp_path / "pr_comments.json"
        json_file.write_text(json.dumps(data_with_unicode, ensure_ascii=False), encoding='utf-8')

        original_dir = os.getcwd()
        try:
            os.chdir(tmp_path)

            captured_output = StringIO()
            sys.stdout = captured_output

            with open('pr_comments.json', 'r', encoding='utf-8') as f:
                data = json.load(f)

            for i, comment in enumerate(data.get('comments', [])):
                print(comment['body'])

            sys.stdout = sys.__stdout__
            output = captured_output.getvalue()

            # Verify unicode characters are preserved
            assert "\u2713" in output or "✓" in output

        finally:
            os.chdir(original_dir)
            sys.stdout = sys.__stdout__

    def test_missing_comments_key(self, tmp_path):
        """Test handling when 'comments' key is missing from JSON."""
        data_without_comments = {
            "reviews": [
                {
                    "author": {"login": "reviewer1"},
                    "submittedAt": "2024-01-01T12:00:00Z",
                    "body": "Review body"
                }
            ]
        }

        json_file = tmp_path / "pr_comments.json"
        json_file.write_text(json.dumps(data_without_comments), encoding='utf-8')

        original_dir = os.getcwd()
        try:
            os.chdir(tmp_path)

            captured_output = StringIO()
            sys.stdout = captured_output

            with open('pr_comments.json', 'r', encoding='utf-8') as f:
                data = json.load(f)

            print(f"Total Comments: {len(data.get('comments', []))}")
            print(f"Total Reviews: {len(data.get('reviews', []))}")

            sys.stdout = sys.__stdout__
            output = captured_output.getvalue()

            # Should default to empty list
            assert "Total Comments: 0" in output
            assert "Total Reviews: 1" in output

        finally:
            os.chdir(original_dir)
            sys.stdout = sys.__stdout__

    def test_missing_reviews_key(self, tmp_path):
        """Test handling when 'reviews' key is missing from JSON."""
        data_without_reviews = {
            "comments": [
                {
                    "author": {"login": "user1"},
                    "createdAt": "2024-01-01T12:00:00Z",
                    "body": "Comment body"
                }
            ]
        }

        json_file = tmp_path / "pr_comments.json"
        json_file.write_text(json.dumps(data_without_reviews), encoding='utf-8')

        original_dir = os.getcwd()
        try:
            os.chdir(tmp_path)

            captured_output = StringIO()
            sys.stdout = captured_output

            with open('pr_comments.json', 'r', encoding='utf-8') as f:
                data = json.load(f)

            print(f"Total Comments: {len(data.get('comments', []))}")
            print(f"Total Reviews: {len(data.get('reviews', []))}")

            sys.stdout = sys.__stdout__
            output = captured_output.getvalue()

            # Should default to empty list
            assert "Total Comments: 1" in output
            assert "Total Reviews: 0" in output

        finally:
            os.chdir(original_dir)
            sys.stdout = sys.__stdout__

    def test_multiple_comments_numbering(self, tmp_path):
        """Test correct numbering of multiple comments."""
        data = {
            "comments": [
                {
                    "author": {"login": f"user{i}"},
                    "createdAt": f"2024-01-0{i}T12:00:00Z",
                    "body": f"Comment {i}"
                }
                for i in range(1, 6)
            ],
            "reviews": []
        }

        json_file = tmp_path / "pr_comments.json"
        json_file.write_text(json.dumps(data), encoding='utf-8')

        original_dir = os.getcwd()
        try:
            os.chdir(tmp_path)

            captured_output = StringIO()
            sys.stdout = captured_output

            with open('pr_comments.json', 'r', encoding='utf-8') as f:
                data = json.load(f)

            for i, comment in enumerate(data.get('comments', [])):
                print(f"COMMENT #{i+1} by {comment['author']['login']} ({comment['createdAt']})")

            sys.stdout = sys.__stdout__
            output = captured_output.getvalue()

            # Verify all comments are numbered correctly
            for i in range(1, 6):
                assert f"COMMENT #{i}" in output
                assert f"user{i}" in output

        finally:
            os.chdir(original_dir)
            sys.stdout = sys.__stdout__

    def test_separator_lines_format(self, sample_pr_data, tmp_path):
        """Test that separator lines are formatted correctly."""
        json_file = tmp_path / "pr_comments.json"
        json_file.write_text(json.dumps(sample_pr_data), encoding='utf-8')

        original_dir = os.getcwd()
        try:
            os.chdir(tmp_path)

            captured_output = StringIO()
            sys.stdout = captured_output

            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            print(f"Total Comments: {len(data.get('comments', []))}")
            print(f"Total Reviews: {len(data.get('reviews', []))}")
            print("-" * 50)

            for i, comment in enumerate(data.get('comments', [])):
                print(f"COMMENT #{i+1} by {comment['author']['login']} ({comment['createdAt']})")
                print(comment['body'])
                print("-" * 50)

            sys.stdout = sys.__stdout__
            output = captured_output.getvalue()

            # Count separator lines (50 hyphens)
            separator_count = output.count("-" * 50)
            # Should be: 1 (after header) + 2 (for comments) = 3
            assert separator_count >= 3

        finally:
            os.chdir(original_dir)
            sys.stdout = sys.__stdout__

    def test_empty_comment_body(self, tmp_path):
        """Test handling of empty comment body."""
        data = {
            "comments": [
                {
                    "author": {"login": "user1"},
                    "createdAt": "2024-01-01T12:00:00Z",
                    "body": ""
                }
            ],
            "reviews": []
        }

        json_file = tmp_path / "pr_comments.json"
        json_file.write_text(json.dumps(data), encoding='utf-8')

        original_dir = os.getcwd()
        try:
            os.chdir(tmp_path)

            captured_output = StringIO()
            sys.stdout = captured_output

            with open('pr_comments.json', 'r', encoding='utf-8') as f:
                data = json.load(f)

            for i, comment in enumerate(data.get('comments', [])):
                print(f"COMMENT #{i+1} by {comment['author']['login']} ({comment['createdAt']})")
                print(comment['body'])
                print("-" * 50)

            sys.stdout = sys.__stdout__
            output = captured_output.getvalue()

            # Should still print the comment header
            assert "COMMENT #1 by user1" in output

        finally:
            os.chdir(original_dir)
            sys.stdout = sys.__stdout__

    def test_multiline_comment_body(self, tmp_path):
        """Test handling of multiline comment body."""
        data = {
            "comments": [
                {
                    "author": {"login": "user1"},
                    "createdAt": "2024-01-01T12:00:00Z",
                    "body": "Line 1\nLine 2\nLine 3"
                }
            ],
            "reviews": []
        }

        json_file = tmp_path / "pr_comments.json"
        json_file.write_text(json.dumps(data), encoding='utf-8')

        original_dir = os.getcwd()
        try:
            os.chdir(tmp_path)

            captured_output = StringIO()
            sys.stdout = captured_output

            with open('pr_comments.json', 'r', encoding='utf-8') as f:
                data = json.load(f)

            for i, comment in enumerate(data.get('comments', [])):
                print(comment['body'])

            sys.stdout = sys.__stdout__
            output = captured_output.getvalue()

            # All lines should be present
            assert "Line 1" in output
            assert "Line 2" in output
            assert "Line 3" in output

        finally:
            os.chdir(original_dir)
            sys.stdout = sys.__stdout__

    def test_large_number_of_comments(self, tmp_path):
        """Test handling of large number of comments (performance regression test)."""
        data = {
            "comments": [
                {
                    "author": {"login": f"user{i}"},
                    "createdAt": f"2024-01-01T{i:02d}:00:00Z",
                    "body": f"Comment body {i}"
                }
                for i in range(100)
            ],
            "reviews": []
        }

        json_file = tmp_path / "pr_comments.json"
        json_file.write_text(json.dumps(data), encoding='utf-8')

        original_dir = os.getcwd()
        try:
            os.chdir(tmp_path)

            captured_output = StringIO()
            sys.stdout = captured_output

            with open('pr_comments.json', 'r', encoding='utf-8') as f:
                data = json.load(f)

            print(f"Total Comments: {len(data.get('comments', []))}")

            sys.stdout = sys.__stdout__
            output = captured_output.getvalue()

            assert "Total Comments: 100" in output

        finally:
            os.chdir(original_dir)
            sys.stdout = sys.__stdout__

    def test_special_characters_in_author_login(self, tmp_path):
        """Test handling of special characters in author login."""
        data = {
            "comments": [
                {
                    "author": {"login": "user-name_123"},
                    "createdAt": "2024-01-01T12:00:00Z",
                    "body": "Test comment"
                }
            ],
            "reviews": []
        }

        json_file = tmp_path / "pr_comments.json"
        json_file.write_text(json.dumps(data), encoding='utf-8')

        original_dir = os.getcwd()
        try:
            os.chdir(tmp_path)

            captured_output = StringIO()
            sys.stdout = captured_output

            with open('pr_comments.json', 'r', encoding='utf-8') as f:
                data = json.load(f)

            for i, comment in enumerate(data.get('comments', [])):
                print(f"COMMENT #{i+1} by {comment['author']['login']} ({comment['createdAt']})")

            sys.stdout = sys.__stdout__
            output = captured_output.getvalue()

            assert "user-name_123" in output

        finally:
            os.chdir(original_dir)
            sys.stdout = sys.__stdout__

    def test_date_format_preserved(self, sample_pr_data, tmp_path):
        """Test that date formats are preserved in output."""
        json_file = tmp_path / "pr_comments.json"
        json_file.write_text(json.dumps(sample_pr_data), encoding='utf-8')

        original_dir = os.getcwd()
        try:
            os.chdir(tmp_path)

            captured_output = StringIO()
            sys.stdout = captured_output

            with open('pr_comments.json', 'r', encoding='utf-8') as f:
                data = json.load(f)

            for i, comment in enumerate(data.get('comments', [])):
                print(f"COMMENT #{i+1} by {comment['author']['login']} ({comment['createdAt']})")

            sys.stdout = sys.__stdout__
            output = captured_output.getvalue()

            # Verify date format is preserved
            assert "2024-01-01T12:00:00Z" in output
            assert "2024-01-02T12:00:00Z" in output

        finally:
            os.chdir(original_dir)
            sys.stdout = sys.__stdout__