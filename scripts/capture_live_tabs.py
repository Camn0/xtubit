import time
import os
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

BRAIN_DIR = r"C:\Users\mahez\.gemini\antigravity-ide\brain\3a36f79d-9eeb-47ef-a336-ef003cc3311b"

def capture_tabs():
    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--window-size=1920,1200")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")

    driver = webdriver.Chrome(options=options)
    try:
        print("[*] Navigating to http://localhost:8501 ...")
        driver.get("http://localhost:8501")

        # Wait until container is present
        WebDriverWait(driver, 30).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "[data-testid='stAppViewContainer']"))
        )

        def wait_until_ready(timeout=30):
            start = time.time()
            while time.time() - start < timeout:
                widgets = driver.find_elements(By.CSS_SELECTOR, "[data-testid='stStatusWidget']")
                if not widgets or not widgets[0].is_displayed():
                    # Double check if any skeleton elements remain
                    skeletons = driver.find_elements(By.CSS_SELECTOR, "[data-testid='stSkeleton']")
                    if len(skeletons) == 0:
                        return True
                time.sleep(1)
            return False

        print("[*] Waiting for Streamlit initial execution to complete...")
        wait_until_ready(30)
        time.sleep(3)

        # Tab names and files
        tabs = driver.find_elements(By.CSS_SELECTOR, "[role='tab']")
        print(f"[+] Found {len(tabs)} tabs")

        for idx, tab in enumerate(tabs):
            tab_name = tab.text.strip().replace(" ", "_").replace("&", "and")
            print(f"[*] Clicking Tab {idx+1}: {tab.text} ...")
            tab.click()
            time.sleep(2)
            wait_until_ready(15)
            time.sleep(2)

            file_path = os.path.join(BRAIN_DIR, f"live_tab_{idx+1}.png")
            driver.save_screenshot(file_path)
            print(f"[+] Saved screenshot to: {file_path}")

        print("[+] All tabs captured successfully!")
    finally:
        driver.quit()

if __name__ == "__main__":
    capture_tabs()
