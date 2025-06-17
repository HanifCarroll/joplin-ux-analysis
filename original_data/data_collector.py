#!/usr/bin/env python3
"""
Joplin UX Analysis Data Collector

This script collects user feedback from GitHub, Reddit, and Discourse forums
for the open-source note-taking app Joplin to support UX analysis and feature roadmap planning.

Usage:
    python data_collector.py [--source github|reddit|discourse]
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Set, Optional, Any
import time

import requests
import praw
from github import Github
from dotenv import load_dotenv


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("data_collector.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


class StateManager:
    """Manages collection state for resumability."""

    def __init__(self, state_file: str = "progress.json"):
        self.state_file = state_file
        self.state = self._load_state()

    def _load_state(self) -> Dict[str, Set[str]]:
        """Load existing state from file."""
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, "r") as f:
                    data = json.load(f)
                    # Convert lists back to sets for faster lookup
                    return {k: set(v) for k, v in data.items()}
            except Exception as e:
                logger.warning(f"Failed to load state file: {e}")

        return {
            "github_issues": set(),
            "reddit_threads": set(),
            "discourse_threads": set(),
        }

    def save_state(self):
        """Save current state to file."""
        try:
            # Convert sets to lists for JSON serialization
            data = {k: list(v) for k, v in self.state.items()}
            with open(self.state_file, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save state: {e}")

    def is_collected(self, source: str, item_id: str) -> bool:
        """Check if an item has already been collected."""
        return item_id in self.state.get(source, set())

    def mark_collected(self, source: str, item_id: str):
        """Mark an item as collected."""
        if source not in self.state:
            self.state[source] = set()
        self.state[source].add(item_id)
        self.save_state()


class GitHubCollector:
    """Collects GitHub issues from laurent22/joplin repository."""

    def __init__(self, token: str, state_manager: StateManager):
        try:
            logger.info("Initializing GitHub client...")
            self.github = Github(token)

            # Test the connection
            user = self.github.get_user()
            logger.info(f"Connected to GitHub as: {user.login}")

            # Check rate limit
            rate_limit = self.github.get_rate_limit()
            logger.info(
                f"GitHub rate limit: {rate_limit.core.remaining}/{rate_limit.core.limit}"
            )

            self.repo = self.github.get_repo("laurent22/joplin")
            logger.info(f"Connected to repository: {self.repo.full_name}")

            self.state_manager = state_manager
            self.cutoff_date = datetime.now(timezone.utc) - timedelta(
                days=180
            )  # 6 months
            logger.info(f"Collecting issues since: {self.cutoff_date}")

        except Exception as e:
            logger.error(f"Failed to initialize GitHub client: {e}")
            raise

    def collect_issues(self) -> List[Dict[str, Any]]:
        """Collect top 100 unique GitHub issues based on engagement."""
        logger.info("Starting GitHub issues collection...")

        try:
            # Get issues sorted by reactions
            reactions_issues = self._get_issues_by_reactions()
            # Get issues sorted by comments
            comments_issues = self._get_issues_by_comments()

            # Combine and deduplicate
            all_issues = {}
            for issue in reactions_issues + comments_issues:
                all_issues[issue.number] = issue

            # Take top 100 by engagement score
            sorted_issues = sorted(
                all_issues.values(),
                key=lambda x: x.get_reactions().totalCount + x.comments,
                reverse=True,
            )[:100]

            collected_issues = []
            for issue in sorted_issues:
                if self.state_manager.is_collected("github_issues", str(issue.number)):
                    logger.info(f"Skipping already collected issue #{issue.number}")
                    continue

                issue_data = self._process_issue(issue)
                if issue_data:
                    collected_issues.append(issue_data)
                    self.state_manager.mark_collected(
                        "github_issues", str(issue.number)
                    )
                    logger.info(f"Collected issue #{issue.number}: {issue.title}")

                    # Rate limiting
                    time.sleep(0.5)

            logger.info(f"Collected {len(collected_issues)} new GitHub issues")
            return collected_issues

        except Exception as e:
            logger.error(f"Error collecting GitHub issues: {e}")
            return []

    def _get_issues_by_reactions(self) -> List:
        """Get issues sorted by reaction count."""
        try:
            logger.info("Fetching issues sorted by reactions...")

            # Check rate limit first
            rate_limit = self.github.get_rate_limit()
            logger.info(
                f"GitHub rate limit - Core: {rate_limit.core.remaining}/{rate_limit.core.limit}"
            )

            if rate_limit.core.remaining < 10:
                logger.warning("Low rate limit remaining, waiting...")
                reset_time = rate_limit.core.reset
                wait_time = (
                    reset_time - datetime.now(timezone.utc)
                ).total_seconds() + 60
                if wait_time > 0:
                    logger.info(f"Waiting {wait_time:.0f} seconds for rate limit reset")
                    time.sleep(wait_time)

            issues = self.repo.get_issues(
                state="open",
                sort="reactions-+1",
                direction="desc",
                since=self.cutoff_date,
            )

            filtered_issues = []
            count = 0
            for issue in issues:
                if count >= 100:
                    break
                if not issue.pull_request:
                    filtered_issues.append(issue)
                    count += 1
                    if count % 10 == 0:
                        logger.info(f"Fetched {count} issues by reactions...")

            logger.info(f"Found {len(filtered_issues)} issues by reactions")
            return filtered_issues

        except Exception as e:
            logger.error(f"Error fetching issues by reactions: {e}")
            return []

    def _get_issues_by_comments(self) -> List:
        """Get issues sorted by comment count."""
        try:
            logger.info("Fetching issues sorted by comments...")

            issues = self.repo.get_issues(
                state="open", sort="comments", direction="desc", since=self.cutoff_date
            )

            filtered_issues = []
            count = 0
            for issue in issues:
                if count >= 100:
                    break
                if not issue.pull_request:
                    filtered_issues.append(issue)
                    count += 1
                    if count % 10 == 0:
                        logger.info(f"Fetched {count} issues by comments...")

            logger.info(f"Found {len(filtered_issues)} issues by comments")
            return filtered_issues

        except Exception as e:
            logger.error(f"Error fetching issues by comments: {e}")
            return []

    def _process_issue(self, issue) -> Optional[Dict[str, Any]]:
        """Process a single issue and its comments."""
        try:
            # Get all comments
            comments = []

            # Add the original issue as first comment
            comments.append(
                {
                    "comment_id": 0,  # Issue body doesn't have a comment ID
                    "author": issue.user.login,
                    "created_at": issue.created_at.isoformat(),
                    "body": issue.body or "",
                }
            )

            # Add all comments
            for comment in issue.get_comments():
                comments.append(
                    {
                        "comment_id": comment.id,
                        "author": comment.user.login,
                        "created_at": comment.created_at.isoformat(),
                        "body": comment.body or "",
                    }
                )

            return {
                "source": "GitHub",
                "issue_id": issue.id,
                "issue_number": issue.number,
                "url": issue.html_url,
                "title": issue.title,
                "author": issue.user.login,
                "state": issue.state,
                "labels": [label.name for label in issue.labels],
                "created_at": issue.created_at.isoformat(),
                "comments": comments,
            }

        except Exception as e:
            logger.error(f"Error processing issue #{issue.number}: {e}")
            return None


class RedditCollector:
    """Collects Reddit posts from r/joplinapp subreddit."""

    def __init__(
        self,
        client_id: str,
        client_secret: str,
        user_agent: str,
        state_manager: StateManager,
    ):
        self.reddit = praw.Reddit(
            client_id=client_id, client_secret=client_secret, user_agent=user_agent
        )
        self.state_manager = state_manager
        self.cutoff_timestamp = (datetime.now() - timedelta(days=180)).timestamp()

    def collect_threads(self) -> List[Dict[str, Any]]:
        """Collect top 50 unique Reddit threads."""
        logger.info("Starting Reddit threads collection...")

        try:
            subreddit = self.reddit.subreddit("joplinapp")

            # Get top posts by score and comments
            top_by_score = list(subreddit.top(time_filter="all", limit=100))
            top_by_comments = list(subreddit.top(time_filter="all", limit=100))

            # Filter by date and combine
            all_posts = {}
            for post in top_by_score + top_by_comments:
                if post.created_utc >= self.cutoff_timestamp:
                    all_posts[post.id] = post

            # Sort by engagement and take top 50
            sorted_posts = sorted(
                all_posts.values(), key=lambda x: x.score + x.num_comments, reverse=True
            )[:50]

            collected_threads = []
            for post in sorted_posts:
                if self.state_manager.is_collected("reddit_threads", post.id):
                    logger.info(f"Skipping already collected thread {post.id}")
                    continue

                thread_data = self._process_thread(post)
                if thread_data:
                    collected_threads.append(thread_data)
                    self.state_manager.mark_collected("reddit_threads", post.id)
                    logger.info(f"Collected thread {post.id}: {post.title}")

                    # Rate limiting
                    time.sleep(1)

            logger.info(f"Collected {len(collected_threads)} new Reddit threads")
            return collected_threads

        except Exception as e:
            logger.error(f"Error collecting Reddit threads: {e}")
            return []

    def _process_thread(self, post) -> Optional[Dict[str, Any]]:
        """Process a single Reddit thread and its comments."""
        try:
            # Expand all comments
            post.comments.replace_more(limit=None)

            comments = []

            # Add original post
            comments.append(
                {
                    "comment_id": post.id,
                    "author": str(post.author) if post.author else "[deleted]",
                    "created_at": datetime.fromtimestamp(
                        post.created_utc, timezone.utc
                    ).isoformat(),
                    "body": post.selftext or "",
                }
            )

            # Add all comments recursively
            for comment in post.comments.list():
                if hasattr(comment, "body"):
                    comments.append(
                        {
                            "comment_id": comment.id,
                            "author": (
                                str(comment.author) if comment.author else "[deleted]"
                            ),
                            "created_at": datetime.fromtimestamp(
                                comment.created_utc, timezone.utc
                            ).isoformat(),
                            "body": comment.body,
                        }
                    )

            return {
                "source": "Reddit",
                "thread_id": post.id,
                "url": f"https://reddit.com{post.permalink}",
                "title": post.title,
                "author": str(post.author) if post.author else "[deleted]",
                "score": post.score,
                "num_comments": post.num_comments,
                "created_at": datetime.fromtimestamp(
                    post.created_utc, timezone.utc
                ).isoformat(),
                "comments": comments,
            }

        except Exception as e:
            logger.error(f"Error processing Reddit thread {post.id}: {e}")
            return None


class DiscourseCollector:
    """Collects threads from Joplin Discourse forum."""

    def __init__(self, state_manager: StateManager):
        self.base_url = "https://discourse.joplinapp.org"
        self.state_manager = state_manager
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "Joplin-UX-Analysis-Bot/1.0"})

    def collect_threads(self) -> List[Dict[str, Any]]:
        """Collect top threads from Support and Features categories."""
        logger.info("Starting Discourse threads collection...")

        collected_threads = []

        # Collect from Support category (ID: 5)
        support_threads = self._collect_category_threads(5, "Support", 50)
        collected_threads.extend(support_threads)

        # Collect from Features category (ID: 7)
        features_threads = self._collect_category_threads(7, "Features", 50)
        collected_threads.extend(features_threads)

        logger.info(f"Collected {len(collected_threads)} new Discourse threads")
        return collected_threads

    def _collect_category_threads(
        self, category_id: int, category_name: str, limit: int
    ) -> List[Dict[str, Any]]:
        """Collect threads from a specific category."""
        try:
            # Get category topics
            url = f"{self.base_url}/c/{category_id}.json"
            response = self.session.get(url)
            response.raise_for_status()

            data = response.json()
            topics = data.get("topic_list", {}).get("topics", [])

            # Filter by date and sort by engagement
            cutoff_date = datetime.now(timezone.utc) - timedelta(days=180)
            recent_topics = []

            for topic in topics:
                created_at = datetime.fromisoformat(
                    topic["created_at"].replace("Z", "+00:00")
                )
                if created_at >= cutoff_date:
                    recent_topics.append(topic)

            # Sort by reply count + views
            recent_topics.sort(
                key=lambda x: x.get("reply_count", 0) + (x.get("views", 0) // 10),
                reverse=True,
            )

            collected = []
            for topic in recent_topics[:limit]:
                topic_id = str(topic["id"])

                if self.state_manager.is_collected("discourse_threads", topic_id):
                    logger.info(
                        f"Skipping already collected Discourse thread {topic_id}"
                    )
                    continue

                thread_data = self._process_topic(topic)
                if thread_data:
                    collected.append(thread_data)
                    self.state_manager.mark_collected("discourse_threads", topic_id)
                    logger.info(
                        f"Collected {category_name} thread {topic_id}: {topic['title']}"
                    )

                    # Rate limiting
                    time.sleep(0.5)

            return collected

        except Exception as e:
            logger.error(f"Error collecting {category_name} threads: {e}")
            return []

    def _process_topic(self, topic) -> Optional[Dict[str, Any]]:
        """Process a single Discourse topic and its posts."""
        try:
            topic_id = topic["id"]
            url = f"{self.base_url}/t/{topic_id}.json"

            response = self.session.get(url)
            response.raise_for_status()

            data = response.json()
            posts = data.get("post_stream", {}).get("posts", [])

            comments = []
            for post in posts:
                comments.append(
                    {
                        "comment_id": post["id"],
                        "author": post.get("username", "unknown"),
                        "created_at": post["created_at"],
                        "body": post.get("cooked", "").strip(),
                    }
                )

            return {
                "source": "Discourse",
                "thread_id": topic_id,
                "url": f"{self.base_url}/t/{topic['slug']}/{topic_id}",
                "title": topic["title"],
                "author": topic.get("last_poster_username", "unknown"),
                "reply_count": topic.get("reply_count", 0),
                "views": topic.get("views", 0),
                "created_at": topic["created_at"],
                "comments": comments,
            }

        except Exception as e:
            logger.error(f"Error processing Discourse topic {topic.get('id')}: {e}")
            return None


class DataCollector:
    """Main data collector orchestrating all sources."""

    def __init__(self):
        load_dotenv()
        self.state_manager = StateManager()
        self._validate_environment()

    def _validate_environment(self):
        """Validate required environment variables."""
        required_vars = [
            "REDDIT_CLIENT_ID",
            "REDDIT_CLIENT_SECRET",
            "REDDIT_USER_AGENT",
            "GITHUB_TOKEN",
        ]

        missing = [var for var in required_vars if not os.getenv(var)]
        if missing:
            logger.error(f"Missing required environment variables: {missing}")
            sys.exit(1)

    def collect_github(self):
        """Collect GitHub issues."""
        try:
            collector = GitHubCollector(os.getenv("GITHUB_TOKEN"), self.state_manager)
            issues = collector.collect_issues()
            self._save_data("github_issues.json", issues)
        except Exception as e:
            logger.error(f"GitHub collection failed: {e}")

    def collect_reddit(self):
        """Collect Reddit threads."""
        try:
            collector = RedditCollector(
                os.getenv("REDDIT_CLIENT_ID"),
                os.getenv("REDDIT_CLIENT_SECRET"),
                os.getenv("REDDIT_USER_AGENT"),
                self.state_manager,
            )
            threads = collector.collect_threads()
            self._save_data("reddit_threads.json", threads)
        except Exception as e:
            logger.error(f"Reddit collection failed: {e}")

    def collect_discourse(self):
        """Collect Discourse threads."""
        try:
            collector = DiscourseCollector(self.state_manager)
            threads = collector.collect_threads()
            self._save_data("discourse_threads.json", threads)
        except Exception as e:
            logger.error(f"Discourse collection failed: {e}")

    def _save_data(self, filename: str, data: List[Dict[str, Any]]):
        """Save collected data to JSON file."""
        if not data:
            logger.info(f"No new data to save for {filename}")
            return

        # Load existing data
        existing_data = []
        if os.path.exists(filename):
            try:
                with open(filename, "r", encoding="utf-8") as f:
                    existing_data = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load existing {filename}: {e}")

        # Merge with new data
        existing_data.extend(data)

        # Save updated data
        try:
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(existing_data, f, indent=2, ensure_ascii=False)
            logger.info(
                f"Saved {len(data)} new items to {filename} (total: {len(existing_data)})"
            )
        except Exception as e:
            logger.error(f"Failed to save {filename}: {e}")

    def collect_all(self):
        """Collect data from all sources."""
        logger.info("Starting data collection from all sources...")
        self.collect_github()
        self.collect_reddit()
        self.collect_discourse()
        logger.info("Data collection completed!")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Collect Joplin user feedback data")
    parser.add_argument(
        "--source",
        choices=["github", "reddit", "discourse"],
        help="Collect from specific source only",
    )

    args = parser.parse_args()
    collector = DataCollector()

    if args.source == "github":
        collector.collect_github()
    elif args.source == "reddit":
        collector.collect_reddit()
    elif args.source == "discourse":
        collector.collect_discourse()
    else:
        collector.collect_all()


if __name__ == "__main__":
    main()
