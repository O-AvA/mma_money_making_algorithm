"""
Scrape the first upcoming UFC event and write a CSV with columns:
FIGHTER, OPPONENT, WEIGHTCLASS, DATE

This is a standalone script with no external config.
"""

import csv
import re
from pathlib import Path
from loguru import logger 

import requests
from bs4 import BeautifulSoup

from src.utils.general import get_data_path 

UPCOMING_EVENTS_URL = 'http://ufcstats.com/statistics/events/upcoming'
OUTPUT_CSV = get_data_path('raw') / 'pred_raw.csv' 


def get_soup(url: str) -> BeautifulSoup:
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return BeautifulSoup(resp.content, 'html.parser')


essential_headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                  'AppleWebKit/537.36 (KHTML, like Gecko) '
                  'Chrome/124.0.0.0 Safari/537.36'
}


def get_first_upcoming_event_url() -> str | None:
    soup = get_soup(UPCOMING_EVENTS_URL)
    for a in soup.find_all('a', class_='b-link b-link_style_black'):
        href = a.get('href') or ''
        if '/event-details/' in href:
            return href
        
    return None


def parse_event_date(soup: BeautifulSoup) -> str | None:
    # Look in the info boxes for a 'Date:' line
    for ul in soup.find_all('ul', class_='b-list__box-list'):
        for li in ul.find_all('li'):
            text = li.get_text(' ', strip=True)
            if text.lower().startswith('date:'):
                return re.sub(r'^date:\s*', '', text, flags=re.I).strip()
    return None


def extract_weightclass_from_fight_page(url: str) -> str | None:
    try:
        soup = get_soup(url)
        head = soup.find('div', class_='b-fight-details__fight-head')
        if head:
            wc = head.get_text(' ', strip=True)
            wc = re.sub(r'\bBout\b', '', wc, flags=re.I).strip()
            return wc
    except Exception:
        return None
    return None


def get_upcoming_bouts_with_meta():
    event_url = get_first_upcoming_event_url()
    if not event_url:
        return None, None, []

    soup = get_soup(event_url)
    event_date = parse_event_date(soup) or 'TBD'

    bouts = []
    rows = soup.find_all('tr', class_='b-fight-details__table-row b-fight-details__table-row__hover js-fight-details-click')
    for tr in rows:
        fighter_names = [
            a.get_text(strip=True)
            for a in tr.find_all('a', class_='b-link b-link_style_black')
            if '/fighter-details/' in (a.get('href') or '')
        ]
        wc = None
        for td in tr.find_all('td'):
            if (td.get('data-th') or '').strip().lower() in ('weight class', 'weightclass', 'wclass'):
                text = td.get_text(' ', strip=True)
                wc = text if text else None
                break
        if not wc:
            fight_url = tr.get('data-link') or ''
            if fight_url:
                wc = extract_weightclass_from_fight_page(fight_url)

        if len(fighter_names) >= 2:
            bouts.append({
                'FIGHTER': fighter_names[0],
                'OPPONENT': fighter_names[1],
                'WEIGHTCLASS': wc or 'TBD',
                'DATE': event_date,
            })

    # fallback if no structured rows
    if not bouts:
        fighter_names = [
            a.get_text(strip=True)
            for a in soup.find_all('a', class_='b-link b-link_style_black')
            if '/fighter-details/' in (a.get('href') or '')
        ]
        for a, b in zip(fighter_names[::2], fighter_names[1::2]):
            bouts.append({
                'FIGHTER': a,
                'OPPONENT': b,
                'WEIGHTCLASS': 'TBD',
                'DATE': event_date,
            })

    return event_url, event_date, bouts

def scrape_pred():
    _, _, bouts = get_upcoming_bouts_with_meta()

    if not bouts:
        OUTPUT_CSV.write_text('')
        print('No upcoming event bouts found.')
        return

    with OUTPUT_CSV.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['FIGHTER', 'OPPONENT', 'WEIGHTCLASS', 'DATE'])
        writer.writeheader()
        writer.writerows(bouts)

    print(f'Wrote {len(bouts)} rows to {OUTPUT_CSV.resolve()}')


if __name__ == '__main__':  
    scrape_pred()
