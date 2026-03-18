"""
ChisCode — GitHub Service
=========================
All GitHub API interactions using the user's OAuth token:
  - Create repository
  - Push files (initial commit)
  - Create feature branch + PR for iterations
  - Get repo info / check existence
  - Rollback via revert commit

Uses httpx async client throughout — no PyGithub dependency.
"""
from __future__ import annotations

import base64
from typing import Any

import httpx

from app.core.logging import get_logger

logger = get_logger(__name__)

GITHUB_API = "https://api.github.com"
HEADERS = {
    "Accept":               "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28",
}


class GitHubError(Exception):
    def __init__(self, message: str, status_code: int = 0):
        super().__init__(message)
        self.status_code = status_code


class GitHubService:
    """Stateless GitHub API client — pass token per call."""

    def __init__(self, token: str):
        self._token = token
        self._headers = {**HEADERS, "Authorization": f"Bearer {token}"}

    # ── Internal ──────────────────────────────────────────────

    async def _get(self, path: str) -> dict:
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.get(f"{GITHUB_API}{path}", headers=self._headers)
            if not r.is_success:
                raise GitHubError(f"GET {path} → {r.status_code}: {r.text}", r.status_code)
            return r.json()

    async def _post(self, path: str, body: dict) -> dict:
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.post(f"{GITHUB_API}{path}", headers=self._headers, json=body)
            if not r.is_success:
                raise GitHubError(f"POST {path} → {r.status_code}: {r.text}", r.status_code)
            return r.json()

    async def _put(self, path: str, body: dict) -> dict:
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.put(f"{GITHUB_API}{path}", headers=self._headers, json=body)
            if r.status_code not in (200, 201):
                raise GitHubError(f"PUT {path} → {r.status_code}: {r.text}", r.status_code)
            return r.json()

    async def _patch(self, path: str, body: dict) -> dict:
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.patch(f"{GITHUB_API}{path}", headers=self._headers, json=body)
            if not r.is_success:
                raise GitHubError(f"PATCH {path} → {r.status_code}: {r.text}", r.status_code)
            return r.json()

    # ── User ──────────────────────────────────────────────────

    async def get_authenticated_user(self) -> dict:
        return await self._get("/user")

    # ── Repositories ──────────────────────────────────────────

    async def create_repo(
        self,
        name:        str,
        description: str = "",
        private:     bool = False,
        auto_init:   bool = False,
    ) -> dict:
        """Create a new repository. Returns repo data including html_url."""
        return await self._post("/user/repos", {
            "name":        name,
            "description": description,
            "private":     private,
            "auto_init":   auto_init,
        })

    async def get_repo(self, owner: str, repo: str) -> dict | None:
        """Return repo data or None if not found."""
        try:
            return await self._get(f"/repos/{owner}/{repo}")
        except GitHubError as e:
            if e.status_code == 404:
                return None
            raise

    async def repo_exists(self, owner: str, repo: str) -> bool:
        return await self.get_repo(owner, repo) is not None

    # ── Commits / File Pushes ─────────────────────────────────

    async def get_default_branch(self, owner: str, repo: str) -> str:
        data = await self._get(f"/repos/{owner}/{repo}")
        return data.get("default_branch", "main")

    async def get_branch_sha(self, owner: str, repo: str, branch: str) -> str | None:
        """Get the latest commit SHA on a branch."""
        try:
            data = await self._get(f"/repos/{owner}/{repo}/git/ref/heads/{branch}")
            return data["object"]["sha"]
        except GitHubError as e:
            if e.status_code in (404, 409):  # 404 = no branch, 409 = repo still initializing
                return None
            raise

    async def push_files(
        self,
        owner:          str,
        repo:           str,
        file_tree:      dict[str, str],
        commit_message: str,
        branch:         str = "main",
    ) -> str:
        """
        Push multiple files to a repo in a single commit using the Git Trees API.
        Returns the new commit SHA.

        Works on both empty and non-empty repos.
        """
        # 1. Get current branch tip (None for brand-new empty repo)
        base_sha = await self.get_branch_sha(owner, repo, branch)

        # 2. Create blobs for every file
        blobs = []
        for path, content in file_tree.items():
            blob = await self._post(f"/repos/{owner}/{repo}/git/blobs", {
                "content":  base64.b64encode(content.encode()).decode(),
                "encoding": "base64",
            })
            blobs.append({
                "path":    path,
                "mode":    "100644",
                "type":    "blob",
                "sha":     blob["sha"],
            })

        # 3. Create tree
        tree_body: dict[str, Any] = {"tree": blobs}
        if base_sha:
            # Get the tree SHA of the current commit to build on top of it
            commit_data = await self._get(f"/repos/{owner}/{repo}/git/commits/{base_sha}")
            tree_body["base_tree"] = commit_data["tree"]["sha"]

        tree = await self._post(f"/repos/{owner}/{repo}/git/trees", tree_body)

        # 4. Create commit
        commit_body: dict[str, Any] = {
            "message": commit_message,
            "tree":    tree["sha"],
        }
        if base_sha:
            commit_body["parents"] = [base_sha]

        commit = await self._post(f"/repos/{owner}/{repo}/git/commits", commit_body)
        new_sha = commit["sha"]

        # 5. Update (or create) branch ref
        if base_sha:
            await self._patch(f"/repos/{owner}/{repo}/git/refs/heads/{branch}", {
                "sha":   new_sha,
                "force": False,
            })
        else:
            await self._post(f"/repos/{owner}/{repo}/git/refs", {
                "ref": f"refs/heads/{branch}",
                "sha": new_sha,
            })

        logger.info("Files pushed to GitHub",
                    owner=owner, repo=repo, branch=branch,
                    files=len(file_tree), sha=new_sha[:8])
        return new_sha

    # ── Branches & PRs ────────────────────────────────────────

    async def create_branch(
        self,
        owner:    str,
        repo:     str,
        branch:   str,
        from_sha: str,
    ) -> None:
        """Create a new branch from a commit SHA."""
        await self._post(f"/repos/{owner}/{repo}/git/refs", {
            "ref": f"refs/heads/{branch}",
            "sha": from_sha,
        })

    async def create_pull_request(
        self,
        owner:  str,
        repo:   str,
        title:  str,
        body:   str,
        head:   str,
        base:   str = "main",
    ) -> dict:
        """Open a PR. Returns PR data including html_url."""
        return await self._post(f"/repos/{owner}/{repo}/pulls", {
            "title": title,
            "body":  body,
            "head":  head,
            "base":  base,
        })

    # ── Full Flows ────────────────────────────────────────────

    async def create_repo_and_push(
        self,
        repo_name:      str,
        description:    str,
        file_tree:      dict[str, str],
        commit_message: str,
        private:        bool = False,
    ) -> dict:
        import asyncio

        repo_data = await self.create_repo(
            name=repo_name,
            description=description,
            private=private,
            auto_init=True,
        )
        owner    = repo_data["owner"]["login"]
        repo_url = repo_data["html_url"]

        # Poll until the Git database is ready (not just the repo API)
        await asyncio.sleep(3)
      
        commit_sha = await self.push_files(
            owner=owner,
            repo=repo_name,
            file_tree=file_tree,
            commit_message=commit_message,
            branch="main",
        )

        return {"repo_url": repo_url, "commit_sha": commit_sha, "owner": owner}
    
    async def push_iteration_pr(
        self,
        owner:          str,
        repo:           str,
        branch_name:    str,
        file_tree:      dict[str, str],
        commit_message: str,
        pr_title:       str,
        pr_body:        str,
    ) -> dict:
        """
        Push updated files to a feature branch and open a PR.
        Returns {"pr_url": ..., "commit_sha": ..., "branch": ...}
        """
        base_sha = await self.get_branch_sha(owner, repo, "main")
        if not base_sha:
            raise GitHubError("Cannot create PR — main branch has no commits")

        await self.create_branch(owner, repo, branch_name, base_sha)

        commit_sha = await self.push_files(
            owner=owner,
            repo=repo,
            file_tree=file_tree,
            commit_message=commit_message,
            branch=branch_name,
        )

        pr = await self.create_pull_request(
            owner=owner,
            repo=repo,
            title=pr_title,
            body=pr_body,
            head=branch_name,
            base="main",
        )

        return {"pr_url": pr["html_url"], "commit_sha": commit_sha, "branch": branch_name}
        
