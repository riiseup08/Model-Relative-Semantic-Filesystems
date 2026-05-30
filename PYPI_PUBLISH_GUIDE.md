# PyPI Publishing Guide - pymrsf v0.4.0

## ✅ Build Complete!

Your package has been successfully built and validated:
- ✅ `pymrsf-0.4.0-py3-none-any.whl` (wheel distribution)
- ✅ `pymrsf-0.4.0.tar.gz` (source distribution)
- ✅ Both distributions passed `twine check`

---

## 📝 Prerequisites

Before uploading, you need:

1. **PyPI Account**: Create one at https://pypi.org/account/register/
2. **API Token**: Generate at https://pypi.org/manage/account/token/
   - Go to https://pypi.org/manage/account/token/
   - Click "Add API token"
   - Name: `pymrsf-upload` (or any name)
   - Scope: "Entire account" (or limit to project after first upload)
   - **Save the token** - you won't see it again!

---

## 🧪 Option 1: Test on TestPyPI First (Recommended)

TestPyPI is a separate instance for testing. Highly recommended for first-time publishers!

### 1. Create TestPyPI Account
- Register at: https://test.pypi.org/account/register/
- Generate API token at: https://test.pypi.org/manage/account/token/

### 2. Upload to TestPyPI
```powershell
c:/Users/pokam/mrsf/venv/Scripts/python.exe -m twine upload --repository testpypi dist/*
```

When prompted:
- Username: `__token__`
- Password: `pypi-...` (your TestPyPI API token)

### 3. Test Installation from TestPyPI
```powershell
# In a fresh venv or conda environment
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ pymrsf
```

Note: `--extra-index-url` allows installing dependencies from real PyPI.

### 4. Verify Installation
```powershell
python -c "import pymrsf; print(pymrsf.__version__)"
```

---

## 🚀 Option 2: Upload to Real PyPI

**⚠️ WARNING:** Once uploaded to PyPI, you **cannot** delete or re-upload the same version number. Make sure everything is correct!

### Upload Command
```powershell
c:/Users/pokam/mrsf/venv/Scripts/python.exe -m twine upload dist/*
```

When prompted:
- Username: `__token__`
- Password: `pypi-...` (your PyPI API token)

### Alternative: Using .pypirc Config File

Create `C:\Users\pokam\.pypirc`:
```ini
[pypi]
username = __token__
password = pypi-YOUR_TOKEN_HERE

[testpypi]
username = __token__
password = pypi-YOUR_TESTPYPI_TOKEN_HERE
```

**⚠️ Security:** Make sure this file is readable only by you!

Then upload without prompts:
```powershell
# TestPyPI
c:/Users/pokam/mrsf/venv/Scripts/python.exe -m twine upload --repository testpypi dist/*

# Real PyPI
c:/Users/pokam/mrsf/venv/Scripts/python.exe -m twine upload dist/*
```

---

## ✅ After Publishing

### 1. Verify on PyPI
- Visit: https://pypi.org/project/pymrsf/
- Check that README renders correctly
- Verify all metadata (author, license, etc.)

### 2. Test Installation
```powershell
pip install pymrsf
```

### 3. Try Different Installation Options
```powershell
# Core only
pip install pymrsf

# With local model support
pip install pymrsf[local]

# With OpenAI support
pip install pymrsf[openai]

# Everything
pip install pymrsf[all]
```

### 4. Tag the Release in Git
```bash
git tag v0.4.0
git push origin v0.4.0
```

### 5. Create GitHub Release
- Go to https://github.com/riiseup08/mrsf/releases/new
- Tag: `v0.4.0`
- Title: `pymrsf v0.4.0 - Multi-Provider RAG Scoring`
- Description: Copy from [CHANGELOG.md](CHANGELOG.md)
- Attach: `dist/pymrsf-0.4.0.tar.gz` and `dist/pymrsf-0.4.0-py3-none-any.whl`

---

## 🔧 Troubleshooting

### "The user isn't allowed to upload to project"
- You need to create the project first (first upload creates it)
- Or get added as a maintainer by the project owner

### "File already exists"
- You've already uploaded this version
- Increment the version in `pyproject.toml` and rebuild:
  ```powershell
  # Edit pyproject.toml: version = "0.4.1"
  c:/Users/pokam/mrsf/venv/Scripts/python.exe -m build
  ```

### "Invalid or non-existent authentication"
- Make sure username is exactly `__token__` (with double underscores)
- Check your API token is correct
- Token should start with `pypi-`

### README not rendering on PyPI
- Check README.md markdown syntax
- Verify `readme = "README.md"` in pyproject.toml
- TestPyPI can help catch this before real PyPI

---

## 📊 Package Stats

After publishing, monitor your package:

- **Downloads**: https://pypistats.org/packages/pymrsf
- **PyPI Page**: https://pypi.org/project/pymrsf/
- **Issues**: https://github.com/riiseup08/mrsf/issues

---

## 🎉 Ready to Publish!

Your package is **verified and ready**. Choose your publishing path:

1. **First-time publisher?** → Start with TestPyPI
2. **Experienced?** → Go straight to PyPI
3. **Want to be extra safe?** → TestPyPI → wait a day → real PyPI

Good luck! 🚀
