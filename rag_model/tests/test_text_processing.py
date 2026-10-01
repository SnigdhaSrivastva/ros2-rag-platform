import pytest

from text_processing import build_prompt, chunk_text, collect_contexts, same_domain_links


class TestChunkText:
    def test_empty_and_whitespace_only_text_returns_no_chunks(self):
        assert chunk_text('') == []
        assert chunk_text('   \n\t  ') == []

    def test_short_text_is_a_single_chunk(self):
        chunks = chunk_text('hello world', chunk_size=100, overlap=10)
        assert [(c.text, c.chunk_id) for c in chunks] == [('hello world', 0)]

    def test_whitespace_is_normalized(self):
        chunks = chunk_text('a  b\n\nc\td', chunk_size=100, overlap=10)
        assert chunks[0].text == 'a b c d'

    def test_chunks_respect_size_and_overlap(self):
        text = ''.join(str(i % 10) for i in range(250))
        chunks = chunk_text(text, chunk_size=100, overlap=20)

        assert all(len(c.text) <= 100 for c in chunks)
        for prev, nxt in zip(chunks, chunks[1:]):
            assert prev.text[-20:] == nxt.text[:20]

    def test_chunks_cover_the_whole_text(self):
        text = ''.join(chr(ord('a') + i % 26) for i in range(1000))
        chunks = chunk_text(text, chunk_size=300, overlap=50)

        rebuilt = chunks[0].text + ''.join(c.text[50:] for c in chunks[1:])
        assert rebuilt == text

    def test_chunk_ids_are_sequential(self):
        chunks = chunk_text('x' * 1000, chunk_size=100, overlap=10)
        assert [c.chunk_id for c in chunks] == list(range(len(chunks)))

    @pytest.mark.parametrize('chunk_size, overlap', [(0, 0), (-5, 0), (100, 100), (100, 150), (100, -1)])
    def test_invalid_parameters_raise(self, chunk_size, overlap):
        with pytest.raises(ValueError):
            chunk_text('some text', chunk_size=chunk_size, overlap=overlap)


class TestBuildPrompt:
    def test_includes_question_and_all_contexts(self):
        prompt = build_prompt('How do I launch Nav2?', ['ctx one', 'ctx two'])
        assert 'Question: How do I launch Nav2?' in prompt
        assert 'ctx one\n\nctx two' in prompt

    def test_instructs_model_to_stay_grounded(self):
        assert 'ONLY the provided context' in build_prompt('q?', ['c'])


class TestCollectContexts:
    def test_keeps_non_empty_texts_with_their_sources(self):
        payloads = [
            {'text': ' alpha ', 'url': 'https://docs.nav2.org/a'},
            {'text': 'beta', 'source': 'docs.ros.org'},
            {'text': 'gamma'},
        ]
        contexts, sources = collect_contexts(payloads)
        assert contexts == ['alpha', 'beta', 'gamma']
        assert sources == ['https://docs.nav2.org/a', 'docs.ros.org', 'unknown']

    def test_skips_missing_and_blank_payloads(self):
        contexts, sources = collect_contexts([None, {}, {'text': '   '}, {'text': None}])
        assert contexts == [] and sources == []


class TestSameDomainLinks:
    PAGE = 'https://docs.nav2.org/concepts/index.html'
    DOMAIN = 'docs.nav2.org'

    def test_resolves_relative_links(self):
        links = same_domain_links(self.PAGE, ['../setup.html', 'costmaps.html'], self.DOMAIN)
        assert links == [
            'https://docs.nav2.org/setup.html',
            'https://docs.nav2.org/concepts/costmaps.html',
        ]

    def test_drops_other_domains_and_non_http_schemes(self):
        hrefs = ['https://github.com/ros-planning', 'mailto:team@nav2.org', 'javascript:void(0)']
        assert same_domain_links(self.PAGE, hrefs, self.DOMAIN) == []

    def test_strips_fragments_and_deduplicates(self):
        hrefs = ['costmaps.html#layers', 'costmaps.html#plugins', 'costmaps.html', '#top']
        assert same_domain_links(self.PAGE, hrefs, self.DOMAIN) == [
            'https://docs.nav2.org/concepts/costmaps.html',
            'https://docs.nav2.org/concepts/index.html',
        ]
