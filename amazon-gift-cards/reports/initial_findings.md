# Initial dataset findings

Full-file count: **152,410 reviews**. Original gzip: **12,292,543 bytes**.

All fields were present and non-null in every record. Zero invalid ratings; 49 blank text bodies. Raw data was not changed.

## Columns

| Field | Observed Python type |
|---|---|
| asin | str |
| helpful_vote | int |
| images | list |
| parent_asin | str |
| rating | float |
| text | str |
| timestamp | int |
| title | str |
| user_id | str |
| verified_purchase | bool |

## Rating distribution

| Stars | Count | Percent |
|---|---:|---:|
| 1 | 12,326 | 8.09% |
| 2 | 1,873 | 1.23% |
| 3 | 3,271 | 2.15% |
| 4 | 6,692 | 4.39% |
| 5 | 128,248 | 84.15% |
| Total | 152,410 | 100.00% |

## Five example records

First five in file order, not a representative sample. Complete raw fields are retained in local `examples.json` (excluded from Git). Excerpts below are shortened for readability.

| Record | Rating | Title | Text excerpt |
|---|---:|---|---|
| 1 | 5.0 | Great gift | Having Amazon money is always good. |
| 2 | 5.0 | amazon gift card | Always the perfect gift.  I have never given one and had someone seem or act disappointed.  Just the opposite.  They are thrilled and excited to have … |
| 3 | 5.0 | perfect gift | When you have a person who is hard to shop for.. an amazon gift card is P E R F E C T.  Man or woman...  No matter what their hobby... lifestyle.. or … |
| 4 | 5.0 | Nice looking | The tin is a nice touch and pretty large.  It's about 4&#34; in diameter and about 1/2&#34; thick.  I added a pretty red ribbon and it is perfect.  Wh… |
| 5 | 1.0 | Not $10 Gift Cards | I bought this pack of Starbucks Gift cards in 2019. Ive given them to friends and I gave 2 to my daughter.<br />My daughter used one recently and it h… |

## Issues and implications

- The attached web extraction showed binary garbage/altered HTML. It was not used for analysis; the original gzip was downloaded and parsed directly.
- Five-star reviews dominate. Accuracy alone will be misleading for later evaluation.
- Blank review bodies require an explicit handling rule; no records have been dropped.
- Raw examples contain HTML tags/entities. Define minimal, auditable normalization next.
- Explicit star statements in titles/bodies can leak labels even after excluding the rating field. No inference is implemented.
- The first records include repeated users. Design duplicate/group-aware sampling and evaluation before splitting.
- Interpreting timestamps as Unix milliseconds gives bounds 2008-08-06T01:28:26+00:00 to 2023-09-06T21:27:21.743000+00:00. Raw integers are preserved.
- Hermes endpoint/model compatibility remains to be verified securely before classification. No credentials were read.

## Verification

Two automated tests passed. Four report artifacts were byte-identical on a repeat full-data run. Rating counts reconcile to the full observation count; exactly five example records were saved.

## Source fingerprint

`e03a258ebd7b1e2591b09862aab65d3dcde9c300744d800aa1fbaf374b7c2340`

See `manifest.json` for download provenance and runtime information.
