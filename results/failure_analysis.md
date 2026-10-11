# Failure analysis (pilot, 30 test queries)

Per-query nDCG@10 is in `per_query.csv`. Below are the largest BM25 vs dense gaps, each with the mechanism checked against the actual top results, not guessed. Chat is quoted only in short fragments, with no usernames.

## By query type (nDCG@10, test)

| type (n) | BM25 | dense | hybrid | hybrid+boost |
|---|---|---|---|---|
| exact (8) | **0.754** | 0.633 | 0.769 | 0.657 |
| event (7) | **0.467** | 0.214 | 0.469 | 0.466 |
| paraphrase (8) | 0.582 | 0.646 | **0.708** | 0.576 |
| slang (7) | 0.748 | **0.858** | 0.847 | 0.814 |

Each cell averages 7–8 queries. Read these as directions, not measurements.

## Where BM25 beats dense

**"drum gun" (exact): BM25 0.94, dense 0.32.** The relevant window has 172 messages, and exactly one says "THE DRUM GUN IS REALLY GOOD". Truncation isn't the cause: the message sits about 42 tokens into a 612-token document, well inside the 512-token limit. The cause is **dilution**. The window embedding is dominated by "sold", "lol" and "???" spam, so a single rare phrase barely moves it. Dense ranked a 147-message "OOOO" hype window first. BM25 needs only one occurrence of a rare term, because IDF does the rest.

**"someone survives russian roulette" (event): BM25 0.41, dense 0.00.** ohnepixel's chat has two different bot games: point roulette ("!roulette 500 … won 500 points") and Russian roulette ("!rr … The trigger is pulled. X survives!"). Dense ranked three point-roulette windows on top, because semantically "roulette" is the strongest signal and the model can't tell the two games apart. BM25 matched "survives", which is rare and decisive.

**"debate over which League champion is the most attractive" (paraphrase): BM25 0.50, dense 0.00.** One question ("who's the hottest LoL champion?") plus two replies, inside a 15-message window about other things. Dense preferred windows talking about "most played champs", where "most" and "champion" overlap on the surface. Same dilution problem as "drum gun", seen in a quieter window.

**Events overall (BM25 0.47, dense 0.21).** Event queries describe *what happened* ("viewer getting unbanned", "team down 0-2"), but chat reacts with fragments ("HE UNBANNED", "0-2 loo"). When the key token survives in chat, BM25 finds it. Dense has to infer the event from a noisy average of the reaction.

## Where dense beats BM25

**"is dantes going to teach jynxzi how to play" (paraphrase): BM25 0.00, dense 1.00.** Chat said "will you coach jynxzi soon". "teach" never appears, and the remaining terms ("dantes", "jynxzi", "play") occur in half the windows of dantes's chat, so BM25 ranked noise. Dense maps teach ≈ coach.

**"six seven meme spam" (slang): BM25 0.11, dense 0.36.** The window is 22× "67" plus variants like "6777777". BM25 can't connect the words "six seven" to the digits. Worse, the meta word **"spam" matched chat literally**: "EVERYONE SPAM R6 MARATHON TONIGHT" ranked first. Dense partly bridges words to digits.

**"W stream" (slang): BM25 0.36, dense 0.79. Self-inflicted.** "stream" is in my stopword list (meant for queries like "what happened on stream"), so BM25 searched for **"w" alone** and ranked a "w n1tr00" window first. The fix is to drop "stream" from the stoplist. I haven't applied it, because the test set has been seen. It should be re-measured on a new eval round.

**"chat complaining about the map choice" (paraphrase): BM25 0.56, dense 0.84.** Chat said "what an ass map who picked ts", "WHY DOES JYNXZI STILL CHOOSING MAPS", "its a bad ranked map". The wording varies a lot, and "complaining" and "choice" never appear. Dense handles the variety. BM25 only gets "map".

## Hybrid

RRF wins overall (0.720, p = 0.03 vs BM25) because the two methods fail on **different** queries. When one method ranks the relevant window high and the other ranks it moderately, fusion keeps it near the top. It loses when one side misses entirely ("is dantes going to teach jynxzi": dense 1.00, hybrid 0.50). BM25's wrong candidates get fused rank credit and push the single correct window down. RRF adds up rank contributions, so a document that one strong list ranks first can lose to documents both lists rank moderately. (The worst-looking hybrid case, "viewer says chat is moving slowly" at 0.05, is confounded: BM25's top window literally says "chat movin slow asf" but was labeled 0.)

## Chat-activity boost

It's neutral for BM25 and dense, and it hurts hybrid (0.720 → 0.640). Dev tuning picked λ = 0. Most pilot queries are **topical** (links, gambling bots, a pun, an emote), and for those, loudness is noise. The clip check points the other way: boosting raises the share of top-1 results that land on a viewer-clipped moment (hybrid 0.53 → 0.61, BM25 0.53 → 0.59). The boost moves results toward *big moments*, which these queries mostly didn't ask for. It should be tested on a query set that has more "when did chat react to X" queries before keeping or dropping it.

## Spam waves, raids, emotes

- **Spam waves** ("67" ×22, "0-4 thor" ×22, "WW stream" ×39): they help BM25 when the query names the spammed token (term frequency is high, though BM25 saturates it), and hurt when the query describes the spam in words ("six seven").
- **Emotes** (sadE, xqcL, xqcGoofy, KEKWait): both methods score ≥ 0.79. They're distinctive tokens, the ohnepixel windows are small, and porter + unicode61 lowercases them consistently. "KEKWait" doesn't match "KEKWiggle" in either method, which is correct.
- **Raids:** none happened before the cutoff, so there's no evidence either way.

## Caveats

- **Label noise in both directions.** The human labeled two source windows 0 (p22 and p23). For p22 the actual fantasy-football window ("Schultz… Gesicki off waivers… Pitts") has no 2 in the qrels, so that query's score rests on a single label-1 window. Per protocol, human labels win and weren't edited.
- **judged@10 for dense is 0.64.** A third of dense's top 10 was never labeled and counts as non-relevant, so dense is likely **underestimated** relative to BM25 (0.84).
