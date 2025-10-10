import requests
from bs4 import BeautifulSoup

def scrape_website_text(url):
    """
    Fetches the HTML content from a URL and extracts all the text.
    Now includes a User-Agent header to avoid 403 errors.
    """
    # Define a User-Agent header to mimic a web browser
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    try:
        # Pass the headers with the GET request
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status() # Raise an exception for bad status codes

        soup = BeautifulSoup(response.content, 'html.parser')

        for script_or_style in soup(['script', 'style']):
            script_or_style.decompose()

        text = soup.get_text(separator=' ', strip=True)
        return text

    except requests.exceptions.RequestException as e:
        return f"Error: Could not retrieve the URL. {e}"
    except Exception as e:
        return f"An unexpected error occurred: {e}"

# --- Main execution block ---
if __name__ == "__main__":
    # Using the URL that previously failed
    target_url = "https://www.bidmc.org/centers-and-departments/transplant-institute/non-transplant-hepatobilary-surgery/guidelines-for-post-surgery-activities"
    output_filename = "scraped2_output.txt"
    
    print(f"Scraping text from: {target_url}")
    
    scraped_text = scrape_website_text(target_url)
    
    if "Error:" not in scraped_text:
        try:
            with open(output_filename, 'w', encoding='utf-8') as f:
                f.write(scraped_text)
            print(f"Successfully saved text to '{output_filename}'")
        except IOError as e:
            print(f"Error: Could not write to file '{output_filename}'. {e}")
    else:
        print(scraped_text)