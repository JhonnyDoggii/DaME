# 🗺️ DaME - Easy Geospatial Metadata Enrichment

[![Download DaME](https://img.shields.io/badge/Download-DaME-blue?style=for-the-badge)](https://github.com/JhonnyDoggii/DaME/releases)

## 📖 What is DaME?

DaME stands for Data-Aware Metadata Enricher. This application helps you automatically extract and improve metadata for geospatial data. It works with vector and raster layers inside QGIS, a popular open-source mapping software.

DaME makes it easy to add important details to your geospatial files. Metadata means data about data—information like the source, date, or resolution of map layers. DaME can export this information in common formats like JSON or XML. It can also enhance metadata using AI-powered tools if you choose.

This helps you keep your maps organized, searchable, and useful for sharing or publishing.

## 💻 System Requirements

Before you download and use DaME, make sure your computer meets these basic requirements:

- **Operating System:** Windows 10 or later, macOS 10.15 or later, or a recent Linux distribution
- **Processor:** 2 GHz dual-core or better
- **Memory (RAM):** At least 4 GB (8 GB recommended)
- **Disk Space:** Minimum 500 MB free space
- **Software Needed:** QGIS version 3.10 or higher installed on your computer
- **Internet Connection:** Required for optional AI enhancements and software updates

## 🚀 Getting Started

DaME is designed for users who do not have programming skills. You interact with it through QGIS plugins and a simple graphical interface.

Here’s what you can expect when you use DaME:

- **Automatic metadata extraction:** It reads basic file information from your geospatial layers.
- **Metadata enrichment:** It adds detailed descriptions and other useful data for your maps.
- **Export options:** Save your enriched metadata as JSON or XML files to use them elsewhere.
- **AI-assisted improvement:** Use large language models to refine your metadata for clarity and completeness.
- **Support for many geospatial formats:** Works with common raster and vector file types.

DaME focuses on making metadata work better so you spend less time managing files and more time analyzing maps.

## 📥 Download & Install

To get started with DaME, follow these steps:

1. **Go to the download page:**  
   Visit the releases section here:  
   [Download DaME Releases](https://github.com/JhonnyDoggii/DaME/releases)

2. **Choose your version:**  
   Find the latest release suitable for your computer's operating system. You will usually see installer files or QGIS plugin packages.

3. **Download the file:**  
   Click on the file to download it. Save it to a folder you can easily find, like your Desktop or Downloads folder.

4. **Install DaME:**

   - **For Windows / macOS:**  
     Double-click the downloaded installer file and follow the on-screen steps.

   - **For Linux:**  
     Follow the instructions in the release notes or documentation. This often involves copying plugin files into the QGIS plugin folder.

5. **Open QGIS:**  
   Launch QGIS on your computer.

6. **Enable DaME Plugin:**  
   In QGIS, go to **Plugins > Manage and Install Plugins**. Search for "DaME" and enable it.

7. **Start Enriching Metadata:**  
   Click on the DaME icon or menu item in QGIS and begin using its tools.

If you run into any issues while installing or using DaME, please check the troubleshooting section below or visit the repository’s GitHub page for help.

## 🖥️ How to Use DaME

This section guides you through using DaME for your first metadata enrichment.

### Step 1: Load Your Data

Open QGIS and import the vector or raster layers you want to manage. You can add shapefiles, GeoTIFFs, or other supported formats as usual.

### Step 2: Open DaME Plugin

Click on the DaME plugin button in QGIS. You will see a simple interface with options to extract and enhance metadata.

### Step 3: Extract Metadata

Click the **Extract Metadata** button. DaME will scan your loaded layers and pull key information like layer name, data extent, creation date, and coordinate system.

### Step 4: Review and Enrich

Look over the extracted metadata shown in DaME. You can manually edit fields or use the **AI Enhance** option to let the software suggest improvements. The AI can fill in descriptions and add context.

### Step 5: Export Metadata

Once happy, choose between JSON or XML formats and save the metadata to your computer. You can attach these files to your project or share them with others.

### Step 6: Use Metadata In Your Workflow

Enriched metadata helps improve data sharing, cataloging, or compliance with standards like FAIR or SDI. You can upload export files to metadata catalogs or use them with other geospatial tools.

## ⚙️ Features at a Glance

- Automatic and manual metadata extraction  
- Support for vector formats (e.g., shapefiles, GeoJSON)  
- Support for raster formats (e.g., GeoTIFF)  
- Export metadata in JSON and XML formats  
- Optional large language model (LLM) assistance for description writing  
- Integration with QGIS as a plugin  
- Compatible with metadata standards for spatial data infrastructures  

## 🛠️ Troubleshooting

If DaME does not work as expected, try the following:

- Verify you have the required version of QGIS installed.
- Check that you downloaded the correct installer for your operating system.
- Make sure your internet connection is on if using AI features.
- Restart QGIS after installing the plugin.
- Check for plugin updates via the QGIS plugin manager.
- Visit the GitHub repository issues page for advice and report problems.

## 📂 About the Project

DaME is an open-source tool developed to simplify geospatial metadata management. It aims to help users create richer, more useful metadata without needing special coding skills. The project supports open standards and works in popular GIS tools.

## 🔗 Useful Links

- DaME Releases: [https://github.com/JhonnyDoggii/DaME/releases](https://github.com/JhonnyDoggii/DaME/releases)  
- QGIS Official Site: [https://qgis.org](https://qgis.org)  
- Metadata Standards Info: Search for FAIR data principles and SDI standards online  

## 🏷️ Keywords

ai-, catalog, fair, geoai, geospatial, llm, metadata, raster, sdi, stac, vector