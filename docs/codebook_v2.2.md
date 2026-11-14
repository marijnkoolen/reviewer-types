# Annotation codebook v2.2 (v2.1 + UA-001; v2.1 = v2 + rulings from adjudication batch 2; v2 revised from v1 after adjudication batch 1; v1 was transcribed from docs/Annotation-categories-book_reviews.pdf)

Task: for every sentence of a Dutch book review, assign ZERO OR MORE labels (multi-label). Judge each sentence in the context of the whole review. Use the child labels only (codes below); never use the parent categories alone.

## Labels (code = name)
**CLS = Classification** — Genre of the book (suspense, literary novel, fantasy…), canonisation (classics, real thriller), references to genre standards, or period/literary epoch. If style is mentioned, use SCX instead of CLS.
 - It is a classic for suspense novels. / The book revolutionizes the horror conventions. / It contains many elements of Romantic literature.

**EVA = Reader_response--Evaluation_of_quality** — Any explicit positive or negative evaluation of the book, general (good, bad, interesting) or also referring to emotional response (boring, exciting). If referring to Style (well written) or Content (well structured plot, deep characters) also assign those labels.
 - It was a good novel without a doubt. (EVA) / The book is not very innovative. (EVA) / I hated the fact that it was so scary (EVA + FEE) / I enjoyed the way the characters are described (EVA + STY + NAR) / 'Heerlijk' (delightful) is EVA, not a feeling. / dat ze zich tot de top van vrouwelijke thrillerauteurs mag rekenen (AUT + CLS + EVA) / schattige verhaaltjes (STY + EVA)
 - Negative: "I liked her earlier books" is not about quality of the reviewed book; annotate as OTW.

**FEE = Reader_response--Feelings** — Any expression of feelings and emotions related to the book, both when referring to the content or to personal feelings and desires including the desire to read on. If there is an emotional response to the expectations of the reader being met or not, tag as FEE. If the feeling is clearly positive or negative it should also be annotated EVA (e.g. "Drie intrigerende verhalen" = FEE + EVA).
 - This book hit me right in the heart. (FEE) / It was a sad story. (FEE) / It made me feel sad. (FEE) / I found this novel quite boring (FEE + EVA) / I had a hard time reading it as it was written (FEE + STY) / Ik bleef op mijn honger zitten (FEE + EVA) / Een spannend verhaal (FEE + EVA)
 - Negative: "Ik heb het met zeer veel plezier gelezen!" — pleasure is not a real feeling, just EVA.
 - v2.2: being left puzzled, unsatisfied or with open questions after reading ("blijf je als lezer achter met vragen") is FEE: an emotional response to expectations not being met.
 - Negative (v2.1): "daar heb ik altijd een beetje moeite mee" — having difficulty with / disliking something is a judgement, EVA, not FEE.

   - v2 clarification: compassion, pity, sympathy, missing characters, regret at parting, irritation, disappointment, suspense/tension felt by the reader are FEE (also when directed at characters). Pleasure/enjoyment alone ("met plezier gelezen", "graag gelezen") and plain judgement words are EVA, not FEE.

**IDI = Reader_response--Identification_and_immersion** — Any process of the reader's identification with the book: immersion in the story, identification with the characters, connection of the content of the book with the reader's biography.
 - v2 clarification: IDI is about taking the characters' perspective, being absorbed in the story ('laat je niet meer los', 'je vergeet alles om je heen', characters 'come alive' for the reader), or relating the story to one's own life. An emotional reaction to a character (compassion etc.) is FEE; use both only if the sentence contains both parts.
 - The pain of the husband has become mine. / It made me think about how children have experienced the war. / Het merendeel van het boek was ik dan ook helemaal méé … / Everyone feels lonely from time to time.

**REF = Reader_response--Reflection** — How the book makes the reader think about their own life, society or other aspects of the world OUTSIDE the book, including changes in the reader's perceptions or appreciations. Also use if the reflection is driven by the content/theme of the book, even if the book is not clearly mentioned.
 - v2 narrowing: the reader's mere opinions, speculation, rhetorical questions or inferences about the book itself ('naar mijn idee…', 'wat mij betreft…') are NOT REF; label them by their content (usually EVA, NAR, STR…). REF is rare.
 - Eerlijk gezegd denk ik dat ik dit boek beter gelezen had toen ik het kocht zo'n 10 jaar geleden. / It helps you understand what it is to lose a child.

**CTX = Reader_response--Reading_Context** — Any mention of time, place, medium of reading, reading plans, order of reading, read/not read, prepublication reading. Expectations for the book caused by other parts of a series or other books of the author are also here. Including the context in which they bought the book. Translation/edition information is MET, not CTX.
 - I read the book in three days. / I read it a few years ago for the first time, then I read it again for a reading group. / I mostly read it on the train, on a tablet. / I received a book from the publisher. / Na haar debuut Zus, … waren mijn verwachtingen hooggespannen …

**RCP = Reader_response--Reception** — Reception by others, official reception, prizes.
 - Negative (v2.1): the reviewer hoping for or predicting a prize ('Het wachten is op een mooie prijs') only expresses the reviewer's own opinion: EVA, not RCP.
 - v2.1: a blurb or quote by someone else on the cover ('Op de achterzijde laat X weten dat…') is RCP + COO.
 - I don't understand why everybody criticized this book. / It was awarded best Sci-fi in 2023. / In another review I read …
 - (Quotes from other reviews belong here, not in QUO.)

**NAR = Content--Narrative** — Narrative features such as plot, characters, emotion of characters, setting.
 - It is the story of two kids who encounter several obstacles. / The story is set during WWII. / The main character experiences loneliness and defeat.
 - "Drie intrigerende verhalen (Three stories)" should NOT be style/structure unless more explicitly noticed (e.g. "Three stories that are told one after another").

**QUO = Content--Quote** — All and only direct quotes from the reviewed book (no quotes from reviews; those are RCP).

**THM = Content--Theme** — Topic/subject of the book (when not referring explicitly to the plot), also personal interpretation, or referring to facts of reality or specific period/place/culture. If the sentence has no explicit notion about the theme, do not use. A reference to 'the theme of the book' without an explicit notion of a theme should not be THM, but IDI.
 - The book is about love. / It is a metaphor for father-son relationship. / It took inspiration from the protests which followed George Floyd's death.
 - Negative: "Everyone feels lonely once in a while" (general reflection about life -> IDI).
 - v2.1 NAR vs THM: events, characters and setting OF THE STORY are NAR. Real-world history, period, culture or social issues discussed as a subject (what the book is about, or what the reviewer says about that subject) are THM. Use the review header (genre) and the rest of the review to decide: in fiction, 'the story is set in 17th-century Amsterdam' = NAR; 'the book is about the slave trade of that era' = THM. If the reviewer goes on about the real-world subject beyond the book, that is REF (+ THM).

**COO = Content--Other** — Other non-linguistic features of the book, e.g. illustrations, length (even if number of words is a textual aspect), cover page.
 - The book also contains a map of Middle Earth. (COO) / I bought it because of the cover. (EVA + COO)
 - v2.1: the back cover / blurb as an object ('op de achterflap', 'op de achterzijde') is COO.

**STY = Style--Stylistic_features** — Any reference to stylistic features such as tone, formality, complexity, narrative perspective, focalization. May contain evaluation (then also EVA). Humor is also related to style. Easy-reading / pace / lightness ('las fijn weg', 'luchtig') relates to style (+ EVA).
 - The book is tersely written. (EVA+STY) / The style is bland. (EVA+STY) / Beautifully described characters. (EVA+STY) / Catharina Blaauwendraad schrijft prachtige gedichten. (AUT + EVA + STY: praise of the writing itself is also style) / It is told in third person but from the perspective of the mother. (STY)

**SCX = Style--Context** — Contextualization of style (period, region, author, translation; this list is not exhaustive: any comparison of the style to something outside the book, including other media, counts — 'filmisch geschreven' = SCX + STY). Comments on the quality of a translation ('prachtig vertaald') are SCX (+ EVA); a bare mention of who translated is MET. Be thorough: check each sentence for style comparisons or translation remarks.
 - It resembles the style of Romantic literature. / I do not like the style of Dan Brown. (SCX + EVA)

**STR = Style--Structure** — Formal organization of the book's contents. If it contains reference to narrative features or focalization also tag STY.
 - The book is divided into two main parts and several chapters.
 - v2: jumping between storylines/timelines, ordering of chapters, 'gecomponeerd' (composition) are STR.
 - v2.1: the pacing of how information is revealed ("een mysterie dat langzaam ontrafeld wordt") is STR (+ NAR).
 - Negative: "verrassende ontknoping" (surprising ending) — narrative, not style. "benieuwd naar hoe de drie verhalen zouden samenkomen" — narrative, not structure.

**MET = Metadata (new in v2)** — Bibliographic/edition information: edition or print run ('Normale druk'), identifier/ISBN, price, publisher, translator, page count as a catalogue fact. Not comments on quality or content (those are SCX, EVA, COO).
 - Normale druk. (MET) / Vertaald door Marcel Otten. (MET) / Uitgeverij X, € 19,99. (MET)

**AUT = Author** — General discussion about the author (also addressing the author), biography and contextualization of the author. Do not tag EVA unless it refers to the book reviewed. If there is any mention of the style of the author, use SCX instead of AUT. If other books of the author are listed/mentioned without discussion of the author, use OTW.
 - Vermeer turned this simple story into a great thriller. (AUT + EVA + CLS) / I love Vermeer's books. (AUT) / Suzanne, you did it again! (AUT) / In haar debuut Zus, haar tweede boek Na Delfine… (AUT)

**OTW = Other_works** — Reflection on other works of the author (including mentions of the book's series) or other authors or adaptations. Do not tag EVA unless it refers to the book reviewed.
 - It was inspired by the wonderful novel "Frankenstein" by Mary Shelley. (OTW) / I haven't seen the film yet, but the book was great (OTW + EVA).
 - Negative: "Het verlangen van de egel is één van de beste werkjes van Tellegen" (no mention of specific works) is not OTW.

**REC = Recommendations** — General and specific reading recommendation.
 - Read this book! / Especially recommended for those who vote PVV.


## General rules (v2)
- Union rule: a sentence gets the union of the labels of its parts (e.g. 'de personen gaan goed voor de lezer leven en roepen compassie op' = IDI + FEE + EVA).
- Evaluation: tag EVA only for an explicit good/bad judgement of the reviewed book (or its parts: style, structure, plot). Descriptive intensifiers are not judgements ('spectaculaire acties' describes the narrative = NAR). Praise of the publisher, series initiative or author in general is not EVA of the book. Do not add EVA just because the tone of the review is positive or because a sentence is about the author or other books.
- Reading context vs Reception vs Other works: a mention of what the reader read before / expected from the series is CTX; opinions of others are RCP; other books themselves are OTW.
- A sentence with no content about the book (greeting, thanks, bare heading such as 'Verwachtingen:') gets an empty label list.
- Read each sentence in the context of the whole review; resolve what it refers to from neighbouring sentences.
- Same context for everyone: each review comes with a header (genre, rating, review year), the same information the model trained on these labels will get. Use only the review text and this header. Do not look up the book or use outside knowledge of it (title, plot, author biography) to decide labels; if a sentence is ambiguous without such knowledge, choose the label the text itself best supports.
