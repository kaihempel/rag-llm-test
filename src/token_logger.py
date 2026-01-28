"""
Token Logger Module

Logs LLM token usage to date-specific files for cost monitoring and analysis.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from collections import defaultdict


class TokenLogger:
    """Logs token usage for LLM API calls."""

    def __init__(self, log_dir: str = "./logs"):
        """
        Initialize the token logger.

        Args:
            log_dir: Directory to store log files
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)

    def _get_log_file_path(self, date: Optional[str] = None) -> Path:
        """
        Get the log file path for a specific date.

        Args:
            date: Date string in YYYY-MM-DD format. If None, uses today's date.

        Returns:
            Path to the log file
        """
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")

        filename = f"token_usage_{date}.jsonl"
        return self.log_dir / filename

    def log_usage(
        self,
        provider: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        metadata: Optional[Dict] = None
    ) -> None:
        """
        Log token usage for an LLM request.

        Args:
            provider: LLM provider name (e.g., 'openai', 'anthropic', 'mistral')
            model: Model name used
            input_tokens: Number of input/prompt tokens
            output_tokens: Number of output/completion tokens
            metadata: Optional additional metadata to log
        """
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "provider": provider.lower(),
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        }

        if metadata:
            log_entry["metadata"] = metadata

        log_file = self._get_log_file_path()

        # Append to JSONL file (one JSON object per line)
        with open(log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

    def get_all_log_files(self) -> List[Path]:
        """
        Get all token usage log files.

        Returns:
            List of paths to log files, sorted by date (newest first)
        """
        log_files = list(self.log_dir.glob("token_usage_*.jsonl"))
        return sorted(log_files, reverse=True)

    def read_log_file(self, log_file: Path) -> List[Dict]:
        """
        Read and parse a log file.

        Args:
            log_file: Path to the log file

        Returns:
            List of log entries
        """
        entries = []

        if not log_file.exists():
            return entries

        with open(log_file, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        # Skip invalid lines
                        continue

        return entries

    def get_statistics(self, log_file: Optional[Path] = None) -> Dict:
        """
        Get usage statistics from a log file or all log files.

        Args:
            log_file: Specific log file to analyze. If None, analyzes all files.

        Returns:
            Dictionary with statistics grouped by provider
        """
        if log_file:
            log_files = [log_file]
        else:
            log_files = self.get_all_log_files()

        # Initialize statistics structure
        stats = {
            "total": {
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "requests": 0
            },
            "by_provider": defaultdict(lambda: {
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "requests": 0,
                "by_model": defaultdict(lambda: {
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                    "requests": 0
                })
            })
        }

        # Process all entries
        for log_file_path in log_files:
            entries = self.read_log_file(log_file_path)

            for entry in entries:
                provider = entry.get("provider", "unknown")
                model = entry.get("model", "unknown")
                input_tokens = entry.get("input_tokens", 0)
                output_tokens = entry.get("output_tokens", 0)
                total_tokens = entry.get("total_tokens", input_tokens + output_tokens)

                # Update total stats
                stats["total"]["input_tokens"] += input_tokens
                stats["total"]["output_tokens"] += output_tokens
                stats["total"]["total_tokens"] += total_tokens
                stats["total"]["requests"] += 1

                # Update provider stats
                stats["by_provider"][provider]["input_tokens"] += input_tokens
                stats["by_provider"][provider]["output_tokens"] += output_tokens
                stats["by_provider"][provider]["total_tokens"] += total_tokens
                stats["by_provider"][provider]["requests"] += 1

                # Update model stats
                stats["by_provider"][provider]["by_model"][model]["input_tokens"] += input_tokens
                stats["by_provider"][provider]["by_model"][model]["output_tokens"] += output_tokens
                stats["by_provider"][provider]["by_model"][model]["total_tokens"] += total_tokens
                stats["by_provider"][provider]["by_model"][model]["requests"] += 1

        # Convert defaultdicts to regular dicts for easier serialization
        stats["by_provider"] = {
            provider: {
                **provider_stats,
                "by_model": dict(provider_stats["by_model"])
            }
            for provider, provider_stats in stats["by_provider"].items()
        }

        return stats

    def get_cost_estimate(self, provider: str, model: str, input_tokens: int, output_tokens: int) -> Optional[float]:
        """
        Estimate cost for token usage (approximate pricing).

        Args:
            provider: Provider name
            model: Model name
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens

        Returns:
            Estimated cost in USD, or None if pricing unknown
        """
        # Pricing as of January 2025 (per 1M tokens)
        # These are approximate and should be updated regularly
        pricing = {
            "openai": {
                "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
                "gpt-4": {"input": 30.00, "output": 60.00},
                "gpt-4-turbo": {"input": 10.00, "output": 30.00},
                "gpt-4o": {"input": 2.50, "output": 10.00},
                "gpt-4o-mini": {"input": 0.15, "output": 0.60},
            },
            "anthropic": {
                "claude-3-5-sonnet-20241022": {"input": 3.00, "output": 15.00},
                "claude-3-opus-20240229": {"input": 15.00, "output": 75.00},
                "claude-3-sonnet-20240229": {"input": 3.00, "output": 15.00},
                "claude-3-haiku-20240307": {"input": 0.25, "output": 1.25},
            },
            "mistral": {
                "mistral-large-latest": {"input": 3.00, "output": 9.00},
                "mistral-medium-latest": {"input": 2.70, "output": 8.10},
                "mistral-small-latest": {"input": 0.20, "output": 0.60},
            }
        }

        provider = provider.lower()
        if provider not in pricing or model not in pricing[provider]:
            return None

        model_pricing = pricing[provider][model]
        input_cost = (input_tokens / 1_000_000) * model_pricing["input"]
        output_cost = (output_tokens / 1_000_000) * model_pricing["output"]

        return input_cost + output_cost


# Global logger instance
_logger_instance: Optional[TokenLogger] = None


def get_logger(log_dir: str = "./logs") -> TokenLogger:
    """
    Get or create the global token logger instance.

    Args:
        log_dir: Directory to store log files

    Returns:
        TokenLogger instance
    """
    global _logger_instance

    if _logger_instance is None:
        _logger_instance = TokenLogger(log_dir)

    return _logger_instance
