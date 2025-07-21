# Documentation Deployment Guide

This documentation site uses Jekyll with the "Just The Docs" theme and is deployed using GitHub Actions for full control over the build environment.

## GitHub Pages Setup

To enable GitHub Pages deployment with Actions:

1. **Go to your repository Settings**
2. **Navigate to Pages section** (under "Code and automation")
3. **Change the Source** from "Deploy from a branch" to "GitHub Actions"
4. **Commit and push** any changes to the main branch

The GitHub Actions workflow (`.github/workflows/jekyll.yml`) will automatically:
- Build the Jekyll site with the Just The Docs theme
- Deploy to GitHub Pages
- Handle all dependencies and plugins

## Local Development

To run the documentation locally:

```bash
cd docs
bundle install
bundle exec jekyll serve --host 0.0.0.0 --port 4000
```

The site will be available at `http://localhost:4000`

## Theme Features

The Just The Docs theme provides:
- **Search functionality** - Built-in site search
- **Responsive design** - Mobile-friendly layout
- **Navigation structure** - Automatic navigation from front matter
- **SEO optimization** - Built-in SEO tags and sitemap
- **Customizable** - Easy to customize colors, fonts, and layout

## Adding New Pages

When adding new documentation pages:

1. Create a new `.md` file in the `docs/` directory
2. Add front matter with navigation order:

```yaml
---
layout: default
title: Your Page Title
nav_order: 10
has_children: false
description: "Page description for SEO"
permalink: /your-page/
---
```

3. The page will automatically appear in the navigation

## Troubleshooting

### Build Failures
- Check the Actions tab in your GitHub repository for build logs
- Ensure all dependencies are properly specified in `Gemfile`
- Verify that front matter syntax is correct in all `.md` files

### Theme Not Loading
- Confirm GitHub Pages is set to use "GitHub Actions" as the source
- Check that the workflow file exists at `.github/workflows/jekyll.yml`
- Verify the theme version in `Gemfile` is compatible

## Benefits of GitHub Actions Deployment

✅ **Full Jekyll version control** - Use any Jekyll version, not just GitHub Pages defaults  
✅ **Custom themes** - Use any Jekyll theme, including Just The Docs  
✅ **Custom plugins** - Install and use any Jekyll plugins  
✅ **Build transparency** - Full build logs and error reporting  
✅ **Caching** - Faster builds with dependency caching  
✅ **Environment control** - Complete control over the build environment  

For more information, see the [Jekyll GitHub Actions documentation](https://jekyllrb.com/docs/continuous-integration/github-actions/).