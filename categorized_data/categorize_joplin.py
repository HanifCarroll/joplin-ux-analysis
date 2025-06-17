#!/usr/bin/env python3
"""
Joplin Feedback Categorization Script
=====================================

Categorizes user feedback from Joplin note-taking app using OpenAI's o4-mini model.
Processes pain points and feature requests with resumable batch processing.

Requirements:
- openai>=1.0.0
- python-dotenv
- API key in .env file as OPENAI_API_KEY=your_key_here

Usage:
    python joplin_categorizer.py

Features:
- Resumable processing with checkpoints
- Batch processing (25 items per API call)
- Cost tracking and estimation
- Cross-analysis of categories vs sentiment/priority/source
- Comprehensive error handling and logging
"""

import json
import os
import time
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict, Counter
import re

from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("categorization.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


class FeedbackCategorizer:
    """Main class for categorizing Joplin user feedback"""

    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = "o4-mini"
        self.batch_size = 25
        self.total_cost = 0.0

        # Cost estimation (approximate rates for o1-mini)
        self.input_cost_per_token = 0.000003  # $3 per 1M tokens
        self.output_cost_per_token = 0.000012  # $12 per 1M tokens

        # File paths
        self.input_file = "analyzed_threads.json"
        self.output_file = "categorized_results.json"
        self.checkpoint_file = "categorization_checkpoint.json"

        # Predefined categories
        self.pain_point_categories = [
            "Sync & Data Management",
            "Editor & Text Input",
            "Mobile Experience",
            "Performance & Speed",
            "UI/UX & Interface",
            "Import/Export & Migration",
            "Search & Organization",
            "Plugin & Extension Issues",
            "Installation & Setup",
            "Markdown & Formatting",
            "File Handling",
            "Backup & Recovery",
            "Cross-Platform Compatibility",
            "Security & Privacy",
            "Other",
        ]

        self.feature_request_categories = [
            "Editor Enhancements",
            "Mobile Features",
            "Sync Improvements",
            "UI/UX Enhancements",
            "Search & Filter Features",
            "Organization & Tagging",
            "Import/Export Features",
            "Plugin & Integration",
            "Performance Optimizations",
            "Security Features",
            "Collaboration Features",
            "Customization Options",
            "Automation & Workflow",
            "Markdown Extensions",
            "Other",
        ]

    def load_checkpoint(self) -> Dict[str, Any]:
        """Load processing checkpoint if exists"""
        if os.path.exists(self.checkpoint_file):
            try:
                with open(self.checkpoint_file, "r", encoding="utf-8") as f:
                    checkpoint = json.load(f)
                logger.info(
                    f"Loaded checkpoint: {checkpoint['processed_items']} items processed"
                )
                return checkpoint
            except Exception as e:
                logger.warning(f"Could not load checkpoint: {e}")

        return {
            "processed_items": 0,
            "categorized_pain_points": [],
            "categorized_feature_requests": [],
            "total_cost": 0.0,
            "last_batch_time": None,
        }

    def save_checkpoint(self, checkpoint: Dict[str, Any]):
        """Save processing checkpoint"""
        try:
            with open(self.checkpoint_file, "w", encoding="utf-8") as f:
                json.dump(checkpoint, f, indent=2, ensure_ascii=False)
            logger.info(
                f"Checkpoint saved: {checkpoint['processed_items']} items processed"
            )
        except Exception as e:
            logger.error(f"Could not save checkpoint: {e}")

    def estimate_tokens(self, text: str) -> int:
        """Rough token estimation (4 chars ≈ 1 token)"""
        return len(text) // 4

    def strip_markdown_json(self, response_text: str) -> str:
        """Strip markdown code blocks from JSON response"""
        # Remove ```json and ``` markers
        response_text = re.sub(r"^```json\s*", "", response_text, flags=re.MULTILINE)
        response_text = re.sub(r"\s*```$", "", response_text, flags=re.MULTILINE)
        return response_text.strip()

    def create_categorization_prompt(self, items: List[str], item_type: str) -> str:
        """Create prompt for categorizing items"""
        categories = (
            self.pain_point_categories
            if item_type == "pain_points"
            else self.feature_request_categories
        )

        prompt = f"""You are analyzing user feedback for the Joplin note-taking application. 
Please categorize the following {item_type.replace('_', ' ')} into the most appropriate categories.

Available categories for {item_type}:
{chr(10).join(f'- {cat}' for cat in categories)}

Items to categorize:
{chr(10).join(f'{i+1}. {item}' for i, item in enumerate(items))}

Please respond with a JSON array where each object has:
- "text": the original text
- "category": the most appropriate category from the list above
- "confidence": "High", "Medium", or "Low" 
- "reasoning": brief explanation (1-2 sentences)

Ensure the response is valid JSON without markdown formatting."""

        return prompt

    def categorize_batch(
        self, items: List[Dict[str, Any]], item_type: str
    ) -> List[Dict[str, Any]]:
        """Categorize a batch of items using OpenAI API"""
        texts = [item["text"] for item in items]
        prompt = self.create_categorization_prompt(texts, item_type)

        # Estimate cost
        input_tokens = self.estimate_tokens(prompt)
        estimated_output_tokens = len(items) * 50  # Rough estimate
        batch_cost = (
            input_tokens * self.input_cost_per_token
            + estimated_output_tokens * self.output_cost_per_token
        )

        logger.info(
            f"Processing batch of {len(items)} {item_type}, estimated cost: ${batch_cost:.4f}"
        )

        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                )

                # Extract and clean response
                response_text = response.choices[0].message.content
                response_text = self.strip_markdown_json(response_text)

                # Parse JSON response
                categorized_items = json.loads(response_text)

                # Add metadata from original items
                for i, cat_item in enumerate(categorized_items):
                    if i < len(items):
                        original_item = items[i]
                        cat_item.update(
                            {
                                "thread_id": original_item["thread_id"],
                                "source": original_item["source"],
                                "sentiment": original_item["sentiment"],
                                "priority": original_item["priority"],
                                "user_type": original_item["user_type"],
                            }
                        )

                # Update cost tracking
                actual_input_tokens = response.usage.prompt_tokens
                actual_output_tokens = response.usage.completion_tokens
                actual_cost = (
                    actual_input_tokens * self.input_cost_per_token
                    + actual_output_tokens * self.output_cost_per_token
                )
                self.total_cost += actual_cost

                logger.info(
                    f"Batch processed successfully. Actual cost: ${actual_cost:.4f}"
                )
                return categorized_items

            except json.JSONDecodeError as e:
                logger.warning(f"JSON parsing error (attempt {attempt + 1}): {e}")
                if attempt == max_retries - 1:
                    logger.error(f"Failed to parse JSON after {max_retries} attempts")
                    raise
                time.sleep(2**attempt)  # Exponential backoff

            except Exception as e:
                logger.warning(f"API error (attempt {attempt + 1}): {e}")
                if attempt == max_retries - 1:
                    logger.error(f"Failed API call after {max_retries} attempts")
                    raise
                time.sleep(2**attempt)  # Exponential backoff

    def process_feedback_items(
        self, threads: List[Dict[str, Any]], checkpoint: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process all feedback items with resumable batching"""

        # Extract all pain points and feature requests
        pain_points = []
        feature_requests = []

        for thread in threads:
            for pain_point in thread.get("pain_points", []):
                pain_points.append(
                    {
                        "text": pain_point,
                        "thread_id": thread["thread_id"],
                        "source": thread["source"],
                        "sentiment": thread["sentiment"],
                        "priority": thread["priority"],
                        "user_type": thread["user_type"],
                    }
                )

            for feature_request in thread.get("feature_requests", []):
                feature_requests.append(
                    {
                        "text": feature_request,
                        "thread_id": thread["thread_id"],
                        "source": thread["source"],
                        "sentiment": thread["sentiment"],
                        "priority": thread["priority"],
                        "user_type": thread["user_type"],
                    }
                )

        logger.info(
            f"Found {len(pain_points)} pain points and {len(feature_requests)} feature requests"
        )

        # Resume from checkpoint
        processed_pain_points = len(checkpoint["categorized_pain_points"])
        processed_feature_requests = len(checkpoint["categorized_feature_requests"])

        # Process pain points
        if processed_pain_points < len(pain_points):
            logger.info(
                f"Processing pain points ({processed_pain_points}/{len(pain_points)} completed)"
            )

            for i in range(processed_pain_points, len(pain_points), self.batch_size):
                batch = pain_points[i : i + self.batch_size]
                batch_num = i // self.batch_size + 1
                total_batches = (
                    len(pain_points) + self.batch_size - 1
                ) // self.batch_size

                logger.info(f"Processing pain points batch {batch_num}/{total_batches}")

                try:
                    categorized_batch = self.categorize_batch(batch, "pain_points")
                    checkpoint["categorized_pain_points"].extend(categorized_batch)
                    checkpoint["processed_items"] += len(batch)
                    checkpoint["total_cost"] = self.total_cost
                    checkpoint["last_batch_time"] = datetime.now().isoformat()

                    self.save_checkpoint(checkpoint)
                    time.sleep(1)  # Rate limiting

                except Exception as e:
                    logger.error(
                        f"Failed to process pain points batch {batch_num}: {e}"
                    )
                    raise

        # Process feature requests
        if processed_feature_requests < len(feature_requests):
            logger.info(
                f"Processing feature requests ({processed_feature_requests}/{len(feature_requests)} completed)"
            )

            for i in range(
                processed_feature_requests, len(feature_requests), self.batch_size
            ):
                batch = feature_requests[i : i + self.batch_size]
                batch_num = i // self.batch_size + 1
                total_batches = (
                    len(feature_requests) + self.batch_size - 1
                ) // self.batch_size

                logger.info(
                    f"Processing feature requests batch {batch_num}/{total_batches}"
                )

                try:
                    categorized_batch = self.categorize_batch(batch, "feature_requests")
                    checkpoint["categorized_feature_requests"].extend(categorized_batch)
                    checkpoint["processed_items"] += len(batch)
                    checkpoint["total_cost"] = self.total_cost
                    checkpoint["last_batch_time"] = datetime.now().isoformat()

                    self.save_checkpoint(checkpoint)
                    time.sleep(1)  # Rate limiting

                except Exception as e:
                    logger.error(
                        f"Failed to process feature requests batch {batch_num}: {e}"
                    )
                    raise

        return checkpoint

    def perform_cross_analysis(
        self,
        categorized_pain_points: List[Dict[str, Any]],
        categorized_feature_requests: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Perform cross-analysis of categories with metadata"""

        def analyze_items(items: List[Dict[str, Any]]) -> Dict[str, Any]:
            sentiment_by_category = defaultdict(lambda: Counter())
            priority_by_category = defaultdict(lambda: Counter())
            source_by_category = defaultdict(lambda: Counter())
            user_type_by_category = defaultdict(lambda: Counter())
            confidence_by_category = defaultdict(lambda: Counter())

            for item in items:
                category = item["category"]
                sentiment_by_category[category][item["sentiment"]] += 1
                priority_by_category[category][item["priority"]] += 1
                source_by_category[category][item["source"]] += 1
                user_type_by_category[category][item["user_type"]] += 1
                confidence_by_category[category][item["confidence"]] += 1

            return {
                "sentiment_by_category": dict(sentiment_by_category),
                "priority_by_category": dict(priority_by_category),
                "source_by_category": dict(source_by_category),
                "user_type_by_category": dict(user_type_by_category),
                "confidence_by_category": dict(confidence_by_category),
            }

        pain_points_analysis = analyze_items(categorized_pain_points)
        feature_requests_analysis = analyze_items(categorized_feature_requests)

        return {
            "pain_points_analysis": pain_points_analysis,
            "feature_requests_analysis": feature_requests_analysis,
            "category_counts": {
                "pain_points": Counter(
                    item["category"] for item in categorized_pain_points
                ),
                "feature_requests": Counter(
                    item["category"] for item in categorized_feature_requests
                ),
            },
        }

    def generate_summary_stats(
        self,
        categorized_pain_points: List[Dict[str, Any]],
        categorized_feature_requests: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Generate summary statistics"""

        def calculate_stats(items: List[Dict[str, Any]]) -> Dict[str, Any]:
            if not items:
                return {}

            return {
                "total_items": len(items),
                "categories": len(set(item["category"] for item in items)),
                "sentiment_distribution": dict(
                    Counter(item["sentiment"] for item in items)
                ),
                "priority_distribution": dict(
                    Counter(item["priority"] for item in items)
                ),
                "source_distribution": dict(Counter(item["source"] for item in items)),
                "user_type_distribution": dict(
                    Counter(item["user_type"] for item in items)
                ),
                "confidence_distribution": dict(
                    Counter(item["confidence"] for item in items)
                ),
                "top_categories": dict(
                    Counter(item["category"] for item in items).most_common(5)
                ),
            }

        return {
            "pain_points": calculate_stats(categorized_pain_points),
            "feature_requests": calculate_stats(categorized_feature_requests),
            "processing_info": {
                "total_cost": self.total_cost,
                "processing_date": datetime.now().isoformat(),
                "model_used": self.model,
                "batch_size": self.batch_size,
            },
        }

    def run(self):
        """Main execution method"""
        start_time = time.time()

        try:
            # Load input data
            logger.info("Loading input data...")
            if not os.path.exists(self.input_file):
                logger.error(f"Input file {self.input_file} not found!")
                return

            with open(self.input_file, "r", encoding="utf-8") as f:
                threads = json.load(f)

            logger.info(f"Loaded {len(threads)} threads from {self.input_file}")

            # Load checkpoint
            checkpoint = self.load_checkpoint()
            self.total_cost = checkpoint["total_cost"]

            # Process feedback items
            logger.info("Starting categorization process...")
            final_checkpoint = self.process_feedback_items(threads, checkpoint)

            # Perform cross-analysis
            logger.info("Performing cross-analysis...")
            cross_analysis = self.perform_cross_analysis(
                final_checkpoint["categorized_pain_points"],
                final_checkpoint["categorized_feature_requests"],
            )

            # Generate summary statistics
            logger.info("Generating summary statistics...")
            summary_stats = self.generate_summary_stats(
                final_checkpoint["categorized_pain_points"],
                final_checkpoint["categorized_feature_requests"],
            )

            # Create final results
            results = {
                "metadata": {
                    "total_cost": self.total_cost,
                    "processing_date": datetime.now().isoformat(),
                    "model_used": self.model,
                    "total_threads": len(threads),
                    "processing_time_minutes": (time.time() - start_time) / 60,
                },
                "categorized_pain_points": final_checkpoint["categorized_pain_points"],
                "categorized_feature_requests": final_checkpoint[
                    "categorized_feature_requests"
                ],
                "cross_analysis": cross_analysis,
                "summary_stats": summary_stats,
            }

            # Save final results
            logger.info(f"Saving results to {self.output_file}...")
            with open(self.output_file, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

            # Clean up checkpoint file
            if os.path.exists(self.checkpoint_file):
                os.remove(self.checkpoint_file)

            # Final summary
            logger.info("=" * 50)
            logger.info("CATEGORIZATION COMPLETE!")
            logger.info(f"Total cost: ${self.total_cost:.4f}")
            logger.info(f"Processing time: {(time.time() - start_time)/60:.1f} minutes")
            logger.info(
                f"Pain points categorized: {len(final_checkpoint['categorized_pain_points'])}"
            )
            logger.info(
                f"Feature requests categorized: {len(final_checkpoint['categorized_feature_requests'])}"
            )
            logger.info(f"Results saved to: {self.output_file}")
            logger.info("=" * 50)

        except KeyboardInterrupt:
            logger.info("Process interrupted by user. Progress saved in checkpoint.")
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            raise


if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not found in environment variables.")
        print("Please create a .env file with: OPENAI_API_KEY=your_key_here")
        exit(1)

    categorizer = FeedbackCategorizer()
    categorizer.run()
