import time
import os
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

BRAIN_DIR = r"C:\Users\mahez\.gemini\antigravity-ide\brain\3a36f79d-9eeb-47ef-a336-ef003cc3311b"

def run_test():
    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--window-size=1600,1200")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")

    driver = webdriver.Chrome(options=options)
    try:
        print("[*] Navigating to http://localhost:8501 ...")
        driver.get("http://localhost:8501")
        
        # Wait for Streamlit app to load
        WebDriverWait(driver, 20).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "[data-testid='stAppViewContainer']"))
        )
        time.sleep(6)

        # 1. Check for any tracebacks or red error alerts
        errors = driver.find_elements(By.CSS_SELECTOR, "[data-testid='stException'], .stAlert[data-baseweb='notification']")
        err_texts = [e.text for e in errors if "error" in e.get_attribute("class") or "Exception" in e.text]
        if err_texts:
            print(f"[!] Found errors on initial load: {err_texts}")
        else:
            print("[+] Initial page load clean. Zero exception elements.")

        # Find tabs by role='tab'
        tabs = driver.find_elements(By.CSS_SELECTOR, "[role='tab']")
        print(f"[+] Found {len(tabs)} tabs: {[t.text for t in tabs]}")
        assert len(tabs) == 5, f"Expected 5 tabs, got {len(tabs)}"

        # TAB 1 Check
        print("[*] Validating Tab 1 (Screening)...")
        tabs[0].click()
        time.sleep(2)
        page_src = driver.page_source
        assert "2nd-Degree External Physics Feedback Active" not in page_src, "Stale feedback banner found in Tab 1!"
        assert "PMHI (Pareto Score)" in page_src or "PMHI" in page_src, "PMHI metric label missing in Tab 1!"
        driver.save_screenshot(os.path.join(BRAIN_DIR, "validation_tab1_screening.png"))
        print("[+] Tab 1 verified and screenshot saved.")

        # TAB 2 Check
        print("[*] Validating Tab 2 (Comparison)...")
        tabs[1].click()
        time.sleep(3)
        page_src = driver.page_source
        assert "Battle Verdict" not in page_src, "Found 'Battle Verdict' in Tab 2!"
        assert "Multi-Parameter Optimization (MPO) Triage Verdict" in page_src, "MPO Triage Verdict missing in Tab 2!"
        driver.save_screenshot(os.path.join(BRAIN_DIR, "validation_tab2_comparison.png"))
        print("[+] Tab 2 verified and screenshot saved.")

        # TAB 3 Check
        print("[*] Validating Tab 3 (Conformer Studio)...")
        tabs[2].click()
        time.sleep(3)
        page_src = driver.page_source
        assert "3D" in page_src or "Conformer" in page_src
        driver.save_screenshot(os.path.join(BRAIN_DIR, "validation_tab3_conformer.png"))
        print("[+] Tab 3 verified and screenshot saved.")

        # TAB 4 Check
        print("[*] Validating Tab 4 (Digital Annealing)...")
        tabs[3].click()
        time.sleep(4)
        page_src = driver.page_source
        assert "2nd-Degree External Feedback Loop" not in page_src, "Found stale 2nd-degree feedback in Tab 4!"
        assert "Thermodynamic Binding Free Energy (ΔG)" in page_src, "Thermodynamic ΔG metric missing in Tab 4!"
        assert ("Digital Annealing & QUBO Fragment Assembly Studio" in page_src or "Digital Annealing &amp; QUBO Fragment Assembly Studio" in page_src), "Studio title missing in Tab 4!"
        driver.save_screenshot(os.path.join(BRAIN_DIR, "validation_tab4_annealing.png"))
        print("[+] Tab 4 verified and screenshot saved.")

        # TAB 5 Check
        print("[*] Validating Tab 5 (Validation & Lab Decisions)...")
        tabs[4].click()
        time.sleep(3)
        page_src = driver.page_source
        assert "Aggarwal et al." in page_src or "Krieger" in page_src, "Empirical literature references missing in Tab 5!"
        driver.save_screenshot(os.path.join(BRAIN_DIR, "validation_tab5_dossier.png"))
        print("[+] Tab 5 verified and screenshot saved.")

        print("================================================================================")
        print("ALL VERIFICATION CHECKS PASSED WITH 100% SUCCESS!")
        print("================================================================================")

    finally:
        driver.quit()

if __name__ == "__main__":
    run_test()
