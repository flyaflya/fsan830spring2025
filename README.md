---
output:
  word_document: default
  html_document: default
  pdf_document: default
---
# fsan830spring2025 - Current Sprint
github repository to support UD's Spring 2025 FSAN830 Business Process Innovation Class
Site: [https://flyaflya.github.io/fsan830spring2025/](https://flyaflya.github.io/fsan830spring2025/)

# Setting Up Git and Making Your First Pull Request

## Part 1: GitHub Account Setup
1. Create a GitHub account at `github.com`
   - Click "Sign up"
   - Choose a username (professional, you'll use it for years)
   - Enter your academic email
   - Create a strong password
   - Verify your email address when prompted

2. Set up your GitHub profile
   - Click your profile icon (top-right)
   - Select "Settings"
   - Add your full name
   - (Optional) Add a profile picture and bio

## Part 2: Software Installation
### Install Cursor IDE
1. Go to `cursor.sh`
2. Click "Download" for your system (Windows/Mac)
3. Run installer and follow prompts

### Install Git
1. Open Cursor terminal (Cmd/Ctrl + Shift + P, type "Terminal")
2. Check if Git is installed: `git --version`
3. If not found:
   - Go to `git-scm.com`
   - Download for your system
   - Run installer with default settings

## Part 3: Git Configuration
In Cursor's terminal:
```bash
git config --global user.name "Your Full Name"
git config --global user.email "your.github@email.com"
```

## Part 4: Creating Your Profile

### Fork Repository
1. Go to `https://github.com/flyaflya/fsan830spring2025`
2. Click "Fork" (top-right)
3. Wait for fork to complete
4. You'll be redirected to your fork

### Clone Your Fork
In Cursor's terminal:
```bash
git clone https://github.com/YOUR-USERNAME/fsan830spring2025
cd fsan830spring2025
```

### Create Branch and Add Files
```bash
git checkout -b update-profile-YOUR-NAME
```

### Add Your Profile Picture
1. Prepare a 300px × 300px image
2. Save as `YOUR-NAME.jpg` (or .png)
3. Place in `images` folder

### Create Profile Page
1. Create `YOUR-NAME.md` in `markdownProfilePages` folder
2. Use template:
```markdown
# Your Name

![Profile Picture](../images/YOUR-NAME.jpg)

## About Me
[Write 2-3 sentences about yourself]

## Research Interests And/Or Favorite Three Topics Covered In Other Classes
- Interest 1
- Interest 2
- Interest 3
```

## Part 5: Submit Changes

### Fixing Profile Issues and Resubmitting
Several profiles need updates. If you see the default horse avatar on the class website, follow these steps:

1. Common issues to fix:
   - Empty or incomplete profile pages
   - Images not properly formatted (must be exactly 300x300 pixels)
   - Images with incorrect aspect ratios (appearing stretched)

2. Update your fork:
   ```bash
   # Switch back to main branch
   git checkout main
   
   # Sync with original repository
   git fetch upstream
   git merge upstream/main
   ```

3. Create new branch for fixes:
   ```bash
   git checkout -b profile-update-YOUR-NAME-v2
   ```

4. Make necessary corrections:
   - For image issues:
     1. First crop image to perfect square (1:1 ratio)
     2. Then resize to exactly 300x300 pixels
   - For profile page:
     1. Ensure content follows template
     2. Check all sections are filled out

5. Commit and push fixes:
   ```bash
   git add .
   git commit -m "Fix profile issues for YOUR-NAME"
   git push origin profile-update-YOUR-NAME-v2
   ```

6. Create new pull request:
   - Go to original repo: `https://github.com/flyaflya/fsan830spring2025`
   - Click "Pull requests" → "New pull request"
   - Click "compare across forks"
   - Select your fork and new branch
   - Reference your previous pull request number in the comment
   - Submit the pull request

Note: If you're having trouble with git commands, you can use VS Code/Cursor's Source Control interface:
- Click Source Control icon (Ctrl+Shift+G)
- Make your changes
- Type commit message
- Click ✓ to commit
- Click "Publish Branch" or "Push"

Replace `YOUR-NAME` with your actual name throughout these instructions.

## Troubleshooting
- If push fails, ensure you're logged into GitHub in terminal:
  ```bash
  git push
  # Follow prompts to log in via browser
  ```
- If images don't show in preview, check file path is correct
- For any errors, copy the exact error message and ask for help
