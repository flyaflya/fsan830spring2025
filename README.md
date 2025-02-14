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

### Commit Your Changes
```bash
git add images/YOUR-NAME.jpg
git add markdownProfilePages/YOUR-NAME.md
git commit -m "Add profile for YOUR-NAME"
git push origin update-profile-YOUR-NAME
```

### Create Pull Request
1. Visit original repo: `https://github.com/flyaflya/fsan830spring2025`
2. You should see a yellow banner suggesting to create a pull request
   - If not, click "Pull requests" → "New pull request"
3. Click "compare across forks"
4. From dropdown menus:
   - base repository: `flyaflya/fsan830spring2025`
   - base: `main`
   - head repository: `YOUR-USERNAME/fsan830spring2025`
   - compare: `update-profile-YOUR-NAME`
5. Click "Create pull request"
6. Add title: "Add profile for YOUR-NAME"
7. Add any comments (optional)
8. Click "Create pull request"

## Troubleshooting
- If push fails, ensure you're logged into GitHub in terminal:
  ```bash
  git push
  # Follow prompts to log in via browser
  ```
- If images don't show in preview, check file path is correct
- For any errors, copy the exact error message and ask for help

Replace `YOUR-NAME` and `YOUR-USERNAME` with your actual name and GitHub username throughout these instructions.
