"""Fetch public THBWiki pages for inspection; never generate dialogue or call an LLM."""
import argparse
import hashlib
import json
import urllib.parse
import urllib.request
from html.parser import HTMLParser
from pathlib import Path


class PlainText(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []
        self.skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in {'script', 'style'}:
            self.skip += 1
        if tag in {'p', 'li', 'h2', 'h3', 'h4', 'br', 'tr'}:
            self.parts.append('\n')

    def handle_endtag(self, tag):
        if tag in {'script', 'style'}:
            self.skip = max(0, self.skip - 1)

    def handle_data(self, text):
        if not self.skip:
            self.parts.append(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('titles', nargs='*')
    parser.add_argument('--pending-eval', action='store_true', help='Fetch still-pending evaluation NPC sources, not dialogue')
    parser.add_argument('--refresh', action='store_true')
    parser.add_argument('--cache', type=Path, default=Path('.cache/canon'))
    args = parser.parse_args()
    args.cache.mkdir(parents=True, exist_ok=True)
    titles = list(args.titles)
    if args.pending_eval:
        manifest = json.loads(Path('data/canon_review_manifest.json').read_text(encoding='utf-8'))
        aliases = {'博丽灵梦': '博丽灵梦/一设资料', '雾雨魔理沙': '雾雨魔理沙/一设资料',
                   '帕秋莉': '帕秋莉·诺蕾姬', '爱丽丝': '爱丽丝·玛格特洛依德',
                   '铃仙': '铃仙·优昙华院·因幡', '蕾蒂': '蕾蒂·霍瓦特洛克',
                   '桑尼·米尔克': '桑尼米尔克', '露娜·切云德': '露娜切露德',
                   '斯塔·萨菲娅': '斯塔萨菲雅',
                   '千亦': '天弓千亦', '日狭美': '豫母都日狭美'}
        titles.extend(aliases.get(name, name) for name in manifest['progress']['eval']['pending_by_npc']
                      if name not in {'天肱', '天独铭'})
    for title in dict.fromkeys(titles):
        target = args.cache / (title.replace('/', '_') + '.json')
        if target.exists() and not args.refresh:
            print(title, 'cached', flush=True)
            continue
        query = urllib.parse.urlencode({'action': 'parse', 'page': title, 'prop': 'text|revid', 'format': 'json', 'redirects': '1'})
        request = urllib.request.Request('https://thwiki.cc/api.php?' + query, headers={'User-Agent': 'Touhou-NPC-canon-audit/1.0'})
        try:
            data = json.load(urllib.request.urlopen(request, timeout=25))
        except Exception as exc:
            print(title, type(exc).__name__, flush=True)
            continue
        if 'error' in data:
            print(title, data['error']['code'], flush=True)
            continue
        page = data['parse']
        html = page['text']['*']
        plain = PlainText()
        plain.feed(html)
        text = '\n'.join(line.strip() for line in ''.join(plain.parts).splitlines() if line.strip())
        record = {'requested_title': title, 'title': page['title'], 'revision': page['revid'], 'url': 'https://thwiki.cc/index.php?' + urllib.parse.urlencode({'title': page['title'], 'oldid': page['revid']}), 'sha256': hashlib.sha256(html.encode()).hexdigest(), 'text': text}
        target.write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        print(page['title'], page['revid'], len(text), flush=True)


if __name__ == '__main__':
    main()
