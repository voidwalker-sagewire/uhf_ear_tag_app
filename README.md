# 🐄 HerdMate: Edge-AI & UHF RFID Cattle Management

**Built by Sagewire Syndicate**

Most AgTech software is built in Silicon Valley boardrooms by people who have never stepped in cow shit. It relies on pristine Wi-Fi, requires you to take off your gloves in the freezing rain, and charges a $500/month subscription for a sterile dashboard.

HerdMate is different. It was built in the cab of a truck, designed for a tablet, and engineered to work at the chute. It is a completely custom, edge-computing hardware bridge that connects rugged UHF RFID scanners, Bluetooth scales, and a Retrieval-Augmented Generation (RAG) Voice AI directly to a private Google Sheet.

No subscriptions. No corporate walled gardens. Just dirt, hardware, and code.

## 🚀 Core Modules

### 1. Field Scanner & Bovine Beacon (`index.html`)
* **UHF RFID Integration:** Designed to work seamlessly with the Chainway C72 (or similar) rugged Android RFID scanners via keyboard wedge or Web Bluetooth.
* **Offline-First:** Scans are queued locally if cell service drops and automatically synced to Google Sheets when you hit a tower.
* **Bovine Beacon:** Enter a specific tag number, and the app uses BLE RSSI signal strength to trigger "Jaws-style" haptic feedback on your device, vibrating faster as you get physically closer to the target animal in the pasture.
* **GPS Logging:** Every scan attaches a geo-pin and real-time local weather data (via OpenWeatherMap).

### 2. DAVE: Digital Agricultural Veterinary Expert (`vet.html` & `herdmate_vet_api.py`)
* **Voice-Activated AI:** Hands covered in mud? Just tap the mic and talk to DAVE. He uses `webkitSpeechRecognition` to listen and a Python backend to talk back.
* **Live Database Context:** DAVE is securely authenticated to your master Google Sheet. If you ask, *"What's wrong with tag 569?"*, DAVE dynamically pulls her birth date, weight, and history before answering.
* **Veterinary RAG Pipeline:** DAVE isn't guessing. The backend includes a ChromaDB vector database (`herdmate_vet_ingest.py`) loaded with actual veterinary literature (MSD Vet Manual, etc.). 
* **Field Triage:** DAVE analyzes symptoms, cross-references your herd history, cites his sources, and reads the recommended treatment protocols out loud via local TTS.

### 3. Smart Calf Scale (`calf-scale.html`)
* **Direct BLE Integration:** Connects directly to generic Bluetooth hanging scales via `navigator.bluetooth`.
* **Live Calibration Engine:** Includes a custom JS calibration multiplier to correct raw hex payloads on the fly using a known hang-weight.
* **Instant Logging:** Parses the BLE payload and writes the birth weight directly to the Google Sheet with one tap.

---

## 🏗 Architecture & Tech Stack
* **Frontend:** Pure HTML/CSS/Vanilla JS. Mobile-first, responsive, and deployable entirely via GitHub Pages (Static).
* **Backend (DAVE):** Python, FastAPI/Flask, ChromaDB (Vector DB for RAG), and PyPDF2/BeautifulSoup for knowledge ingestion.
* **Database:** Google Sheets API (v4). Acts as a highly accessible, infinitely flexible, free database.
* **Auth:** Google Identity Services (OAuth2 for frontend users) and Google Cloud Service Accounts (Server-to-server auth for the Python backend).

---

## 🛠 Setup & Installation

### Phase 1: The Database (Google Sheets)
1. Go to Google Cloud Console and create a new project.
2. Enable the **Google Sheets API** and **Google Drive API**.
3. Create an **OAuth 2.0 Client ID** (Web application). Add your frontend URL to the authorized JavaScript origins.
4. Create a **Service Account** and download the `credentials.json` key (Keep this safe. NEVER push this to GitHub).

### Phase 2: Frontend Configuration (Security First)
Because this is a static frontend, **API keys cannot be hardcoded in the public repository.** 1. Clone this repository.
2. Copy `config.example.js` and rename it to `config.js`.
3. Open `config.js` and add your private API keys:
   ```javascript
   const CONFIG = {
       OWM_KEY: 'your_open_weather_map_api_key',
       MASTER_SHEET_ID: 'your_google_sheet_id' // Found in the URL of your Google Sheet
   };

```
 4. **Note:** .gitignore is set up to block config.js from being committed. Ensure it stays that way!
 5. Host the frontend via GitHub Pages, Vercel, or any static host.
### Phase 3: Backend Configuration (DAVE Vet AI)
 1. Spin up a Linux server (Ubuntu/Debian recommended).
 2. Clone the repo and safely transfer your Service Account credentials.json to the server (e.g., via SSH/SCP).
 3. Install dependencies:
   ```bash
   pip3 install chromadb PyPDF2 sentence-transformers beautifulsoup4 requests google-auth google-api-python-client
   
   ```
 4. Run the ingestion script to build DAVE's brain:
   ```bash
   python3 herdmate_vet_ingest.py --msd
   
   ```
 5. Start the backend API service (herdmate_vet_api.py) and expose it to your frontend domain.
## 📜 License
This project is licensed under the **GNU General Public License v3.0 (GPL-3.0)**. You are free to use, modify, and distribute this software, but any derivative works must also be open-source and use the same license. See the LICENSE file for details.
*Built in the dirt. Coded in the truck. Secured by Sagewire Syndicate.*
```

```
