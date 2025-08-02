#!/usr/bin/env python3
"""
Script to set up GitHub repository topics and configurations.
Requires GitHub Personal Access Token with repo permissions.
"""

import requests
import json
import os
from pathlib import Path

def setup_repository_topics(token, owner, repo):
    """Add topics to the repository."""
    
    # Recommended topics for medical imaging repository
    topics = [
        "medical-imaging",
        "mri-analysis", 
        "tbi",
        "deep-learning",
        "brain-segmentation",
        "neuroscience",
        "python",
        "pytorch",
        "monai",
        "medical-ai",
        "neuroimaging",
        "brain-injury",
        "segmentation",
        "3d-imaging"
    ]
    
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.mercy-preview+json"
    }
    
    url = f"https://api.github.com/repos/{owner}/{repo}/topics"
    
    try:
        response = requests.put(url, headers=headers, json={"names": topics})
        if response.status_code == 200:
            print(f"✅ Successfully added {len(topics)} topics to repository")
            print(f"Topics: {', '.join(topics)}")
        else:
            print(f"❌ Failed to add topics: {response.status_code}")
            print(response.text)
    except Exception as e:
        print(f"❌ Error adding topics: {e}")

def create_release(token, owner, repo, tag, name, body):
    """Create a release for the repository."""
    
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json"
    }
    
    url = f"https://api.github.com/repos/{owner}/{repo}/releases"
    
    release_data = {
        "tag_name": tag,
        "name": name,
        "body": body,
        "draft": False,
        "prerelease": False
    }
    
    try:
        response = requests.post(url, headers=headers, json=release_data)
        if response.status_code == 201:
            print(f"✅ Successfully created release: {tag}")
            return response.json()["upload_url"]
        else:
            print(f"❌ Failed to create release: {response.status_code}")
            print(response.text)
            return None
    except Exception as e:
        print(f"❌ Error creating release: {e}")
        return None

def main():
    """Main function to set up repository."""
    
    print("🚀 TBI Lesion Analysis Pipeline - Repository Setup")
    print("=" * 50)
    
    # Get GitHub token
    token = os.getenv("GITHUB_TOKEN")
    if not token:
        print("❌ Please set GITHUB_TOKEN environment variable")
        print("You can create a Personal Access Token at: https://github.com/settings/tokens")
        print("Required permissions: repo, workflow")
        return
    
    owner = "thildebrant"
    repo = "tbi"
    
    print(f"Setting up repository: {owner}/{repo}")
    print()
    
    # Step 1: Add topics
    print("📝 Step 1: Adding repository topics...")
    setup_repository_topics(token, owner, repo)
    print()
    
    # Step 2: Create initial release
    print("📦 Step 2: Creating initial release...")
    
    release_body = """## Initial Release

This is the initial release of the TBI Lesion Analysis Pipeline.

### Features
- Complete deep learning pipeline for TBI lesion analysis from MRI scans
- Multi-class lesion segmentation (7 lesion types)
- Brain atlas integration with 10 anatomical zones
- Comprehensive preprocessing, segmentation, and quantification modules
- Medical imaging data privacy considerations

### Documentation
- Comprehensive README with usage examples
- Contributing guidelines
- MIT License
- CI/CD pipeline with GitHub Actions

### Note
Model files and sample data are not included in this release due to size constraints.
Please refer to the documentation for training instructions and data requirements.

### Medical Disclaimer
This software is designed for research purposes. Clinical use requires additional validation and regulatory approval.
"""
    
    upload_url = create_release(token, owner, repo, "v1.0.0", "Initial Release", release_body)
    
    if upload_url:
        print("✅ Release created successfully!")
        print(f"📋 Release URL: https://github.com/{owner}/{repo}/releases/tag/v1.0.0")
    print()
    
    # Step 3: Instructions for branch protection
    print("🛡️ Step 3: Branch Protection Setup Instructions")
    print("Please manually configure branch protection rules:")
    print("1. Go to: https://github.com/thildebrant/tbi/settings/branches")
    print("2. Click 'Add rule' for the 'main' branch")
    print("3. Enable the following options:")
    print("   - Require a pull request before merging")
    print("   - Require status checks to pass before merging")
    print("   - Require branches to be up to date before merging")
    print("   - Include administrators")
    print("4. Save changes")
    print()
    
    print("🎉 Repository setup complete!")
    print(f"📖 View your repository: https://github.com/{owner}/{repo}")

if __name__ == "__main__":
    main() 