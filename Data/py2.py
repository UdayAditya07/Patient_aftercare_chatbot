import requests
from bs4 import BeautifulSoup
import warnings

def scrape_website_text(url):
    """
    Fetches the HTML content from a URL and extracts meaningful text.
    Includes SSL bypass for legacy servers (like BIDMC).
    """
    headers = {
        'User-Agent': (
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
            'AppleWebKit/537.36 (KHTML, like Gecko) '
            'Chrome/91.0.4472.124 Safari/537.36'
        )
    }

    try:
        # Suppress SSL warnings
        warnings.filterwarnings("ignore", message="Unverified HTTPS request")

        # Disable SSL verification (needed for BIDMC site)
        response = requests.get(url, headers=headers, timeout=15, verify=False)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, 'html.parser')

        # Remove scripts, styles, and nav elements
        for tag in soup(['script', 'style', 'header', 'footer', 'nav']):
            tag.decompose()

        # Collect only paragraph and list text
        text_elements = [tag.get_text(" ", strip=True) for tag in soup.find_all(['p', 'li'])]
        clean_text = "\n".join(text_elements)

        return clean_text

    except requests.exceptions.RequestException as e:
        return f"Error: Could not retrieve the URL. {e}"
    except Exception as e:
        return f"An unexpected error occurred: {e}"


# --- Main execution block ---
if __name__ == "__main__":
    target_url = "https://www.bidmc.org/centers-and-departments/transplant-institute/non-transplant-hepatobilary-surgery/guidelines-for-post-surgery-activities"
    output_filename = "scraped2_output.txt"

    print(f"Scraping text from: {target_url}")

    scraped_text = scrape_website_text(target_url)

    if "Error:" not in scraped_text:
        try:
            with open(output_filename, 'w', encoding='utf-8') as f:
                f.write(scraped_text)
            print(f"✅ Successfully saved text to '{output_filename}'")
        except IOError as e:
            print(f"Error: Could not write to file '{output_filename}'. {e}")
    else:
        print(scraped_text)
