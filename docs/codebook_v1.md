# Annotation codebook v1 (transcribed from docs/Annotation-categories-book_reviews.pdf)

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

**IDI = Reader_response--Identification_and_immersion** — Any process of the reader's identification with the book: immersion in the story, identification with the characters, connection of the content of the book with the reader's biography.
 - The pain of the husband has become mine. / It made me think about how children have experienced the war. / Het merendeel van het boek was ik dan ook helemaal méé … / Everyone feels lonely from time to time.

**REF = Reader_response--Reflection** — Reflections about (own) life and society, including changes in the reader's perceptions or appreciations. Also use if the reflection is driven by the content/theme of the book, even if the book is not clearly mentioned.
 - Eerlijk gezegd denk ik dat ik dit boek beter gelezen had toen ik het kocht zo'n 10 jaar geleden. / It helps you understand what it is to lose a child.

**CTX = Reader_response--Reading_Context** — Any mention of time, place, medium of reading, reading plans, order of reading, read/not read, prepublication reading. Expectations for the book caused by other parts of a series or other books of the author are also here. Including the context in which they bought the book.
 - I read the book in three days. / I read it a few years ago for the first time, then I read it again for a reading group. / I mostly read it on the train, on a tablet. / I received a book from the publisher. / Na haar debuut Zus, … waren mijn verwachtingen hooggespannen …

**RCP = Reader_response--Reception** — Reception by others, official reception, prizes.
 - I don't understand why everybody criticized this book. / It was awarded best Sci-fi in 2023. / In another review I read …
 - (Quotes from other reviews belong here, not in QUO.)

**NAR = Content--Narrative** — Narrative features such as plot, characters, emotion of characters, setting.
 - It is the story of two kids who encounter several obstacles. / The story is set during WWII. / The main character experiences loneliness and defeat.
 - "Drie intrigerende verhalen (Three stories)" should NOT be style/structure unless more explicitly noticed (e.g. "Three stories that are told one after another").

**QUO = Content--Quote** — All and only direct quotes from the reviewed book (no quotes from reviews; those are RCP).

**THM = Content--Theme** — Topic/subject of the book (when not referring explicitly to the plot), also personal interpretation, or referring to facts of reality or specific period/place/culture. If the sentence has no explicit notion about the theme, do not use. A reference to 'the theme of the book' without an explicit notion of a theme should not be THM, but IDI.
 - The book is about love. / It is a metaphor for father-son relationship. / It took inspiration from the protests which followed George Floyd's death.
 - Negative: "Everyone feels lonely once in a while" (general reflection about life -> IDI).

**COO = Content--Other** — Other non-linguistic features of the book, e.g. illustrations, length (even if number of words is a textual aspect), cover page.
 - The book also contains a map of Middle Earth. (COO) / I bought it because of the cover. (EVA + COO)

**STY = Style--Stylistic_features** — Any reference to stylistic features such as tone, formality, complexity, narrative perspective, focalization. May contain evaluation (then also EVA). Humor is also related to style.
 - The book is tersely written. (EVA+STY) / The style is bland. (EVA+STY) / Beautifully described characters. (EVA+STY) / It is told in third person but from the perspective of the mother. (STY)

**SCX = Style--Context** — Contextualization of style (period, region, author, translation).
 - It resembles the style of Romantic literature. / I do not like the style of Dan Brown. (SCX + EVA)

**STR = Style--Structure** — Formal organization of the book's contents. If it contains reference to narrative features or focalization also tag STY.
 - The book is divided into two main parts and several chapters.
 - Negative: "verrassende ontknoping" (surprising ending) — narrative, not style. "benieuwd naar hoe de drie verhalen zouden samenkomen" — narrative, not structure.

**AUT = Author** — General discussion about the author (also addressing the author), biography and contextualization of the author. Do not tag EVA unless it refers to the book reviewed. If there is any mention of the style of the author, use SCX instead of AUT. If other books of the author are listed/mentioned without discussion of the author, use OTW.
 - Vermeer turned this simple story into a great thriller. (AUT + EVA + CLS) / I love Vermeer's books. (AUT) / Suzanne, you did it again! (AUT) / In haar debuut Zus, haar tweede boek Na Delfine… (AUT)

**OTW = Other_works** — Reflection on other works of the author (including mentions of the book's series) or other authors or adaptations. Do not tag EVA unless it refers to the book reviewed.
 - It was inspired by the wonderful novel "Frankenstein" by Mary Shelley. (OTW) / I haven't seen the film yet, but the book was great (OTW + EVA).
 - Negative: "Het verlangen van de egel is één van de beste werkjes van Tellegen" (no mention of specific works) is not OTW.

**REC = Recommendations** — General and specific reading recommendation.
 - Read this book! / Especially recommended for those who vote PVV.

A sentence may have no label at all (e.g. thanks, greetings, meta-text) — then output an empty list.
