# Cookie Warrior : Product Intelligence Workspace

![GitHub repo size](https://img.shields.io/github/repo-size/KevinJonathan7708/unihack-hackathon-mvp)

# Presentation File
[![Google Slides](https://shields.io)]([YOUR_GOOGLE_SLIDES_URL_HERE](https://docs.google.com/presentation/d/1aIMP835th99UFMHQpl4MgAPKnR2Cl92oVrChp9vztUk/edit?usp=sharing))

<img width="768" height="691" alt="Cappuccino-Assassino-Viral-TikTok-768x691" src="https://github.com/user-attachments/assets/8586ddfd-f921-477c-ab09-9c4080c05ad8" />


## The Live Demo
Experience the live application here: **[UniHack - Product Intelligence · Streamlit](https://unihack-hackathon-mvp-wrpbcvmhlwsg5bdryxfmyw.streamlit.app/)**

---

## The Overview
Welcome to the MVP repository for the UniHack Hackathon! This project provides a robust data enrichment pipeline designed to solve incomplete Product Information Management (PIM) data. 

By leveraging automation in `app_final.py`, the pipeline ingests minimal product identifiers and aggressively enriches them, generating everything from marketing descriptions and SEO-friendly metadata to granular technical attributes (e.g., voltages, dimensions, sound levels) and media URLs.

## Repository Structure
As seen in `image_51a3fe.png`, the core of the repository consists of:
*   **`app_final.py`**: The main application Streamlit script containing the pipeline logic.
*   **`requirements.txt`**: The list of Python dependencies required to run the pipeline.
*   **`README.md`**: Project documentation (this file).

## Dataset Reference
To understand the pipeline's transformation capabilities, we have provided two reference files:

### 1. The Input
**File:** `Unihack_ Sample Dataset - Input (1).csv`
This file represents the raw data starting point. It contains only 6 basic columns:
*   `Mfg_Part_Num`
*   `Part_Desc`
*   `E1_Brand`
*   `Unilog_Brand`
*   `DIB_Brand`
*   `Part_Manuf`

### 2. The Output
**File:** `Unihack_ Expected Output - Delivery Format (1).csv`
This file demonstrates the final, enriched delivery format. The pipeline maps the 6 input columns into a massive schema of over 100 columns, including:
*   **Categorization:** `Dept`, `Class`, `Fine`, `UNSPSC`.
*   **Rich Descriptions:** `MOBILE_DESC`, `INVOICE_DESC`, `SHORT_DESC`, `MARKETING_DESCRIPTION`.
*   **Extracted Features:** Up to 20 individual `ITEM_FEATURES` (e.g., "Adjustable 2nd Rack", "Leak Detection System").
*   **Dynamic Attributes:** 50 dynamically mapped `ATTRIBUTE_LABEL`, `ATTRIBUTE_VALUE`, and `ATTRIBUTE_UOM` (Unit of Measure) fields for precise technical specifications.
*   **Digital Assets:** Image URLs, `MFR URL`, Specification Sheets, and Warranty documents.

## Installation & Setup

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/KevinJonathan7708/unihack-hackathon-mvp.git](https://github.com/KevinJonathan7708/unihack-hackathon-mvp.git)
    cd unihack-hackathon-mvp
    ```

2.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the application locally:**
    ```bash
    streamlit run app_final.py
    ```

## Built With
*   Python 3
*   Streamlit (for the web interface)
*   Pandas (for complex data transformations)

## 💡 Use Case
This MVP is built to automate away the hundreds of manual hours typically required by e-commerce managers and data stewards to clean, standardize, and enrich product catalogs before going to market.


**##App Screenshot**
<img width="1512" height="769" alt="Screenshot 2026-08-21 at 11 57 12 PM" src="https://github.com/user-attachments/assets/7fc10104-483b-4527-b4b5-302dc6ca166f" />
