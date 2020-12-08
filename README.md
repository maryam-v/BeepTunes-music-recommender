# Music Playlist Recommender System

Here we have implemented a music recommender system using [Beeptunes](https://beeptunes.com/) (a large digital music store in Iran) dataset as the final project for [Rahnema College](https://rahnemacollege.com/)’s internship program.

## Roadmap
- Exploratory Data Analysis (EDA)
- Content-Based and Collaborative Filtering
- Hybrid Model
- API
- Example

# 1) Summary of EDA
Number of users: 586785 <br>
Number of albums: 16780 <br>
Number of tracks: 125029 <br>
Number of artists: 16045 <br>
<img src="pics/1.png" alt="drawing" width="500"/><br>
<img src="pics/2.png" alt="drawing" width="500"/><br>

# 2) Content-Based & Collaborative Filtering
- Recommends items with similar content to what the user has already rated positively <br>
- Beeptunes type key data for some of the tracks: <br>
<img src="pics/3.png" alt="drawing" width="400"/><br>
- Computing Cosine Similarity for each type key independently : <br>
<img src="pics/4.png" alt="drawing" width="500"/>


## Similarity Matrix:
- Give each type key a coefficient to compute the complete similarity matrix <br>

𝑆𝑖𝑚𝑖𝑙𝑎𝑟𝑖𝑡𝑦 = 𝑥 1 ∗ 𝑔𝑒𝑛𝑟𝑒𝑆𝑖𝑚𝑖𝑙𝑎𝑟𝑖𝑡𝑦 + 𝑥 2 ∗ 𝑜𝑟𝑐ℎ𝑒𝑠𝑡𝑟𝑎𝑡𝑖𝑜𝑛𝑆𝑖𝑚𝑖𝑙𝑎𝑟𝑖𝑡𝑦 𝑥 3
∗ 𝑐𝑢𝑟𝑎𝑡𝑖𝑜𝑛𝑆𝑖𝑚𝑖𝑙𝑎𝑟𝑖𝑡𝑦 + 𝑥 4 ∗ 𝑚𝑜𝑜𝑑𝑆𝑖𝑚𝑖𝑙𝑎𝑟𝑖𝑡𝑦 + 𝑥 5 ∗ 𝑓𝑜𝑟𝑚𝑆𝑖𝑚𝑖𝑙𝑎𝑟𝑖𝑡 <br>

## Results:
- Calculating Similarity Matrix for all the data is computationally expensive
- Select 85 users with the most interactions (1000 tracks) 
<img src="pics/5.png" alt="drawing" width="300"/>

- Content-based:
<img src="pics/6.png" alt="drawing" width="300"/>

- Non-negative Matrix Factorization:
<img src="pics/7.png" alt="drawing" width="500"/>


# 3) Hybrid Model
- Mix of recommender systems (Parallelized Hybrid)

<img src="pics/8.png" alt="drawing" width="200"/>

# 4) API
- Adding to database
	- Tracks
	- Download/likes/purchase for each track
	- Download/likes/purchase for each album
	- Likes for artists
- Reading from database
	- “Discovery” recommendation

<img src="pics/9.png" alt="drawing" width="500"/>

- Instructions
<img src="pics/10.png" alt="drawing" width="500"/>
<br>
<img src="pics/11.png" alt="drawing" width="500"/>
<br>
<img src="pics/12.png" alt="drawing" width="500"/>
<br>
<img src="pics/13.png" alt="drawing" width="500"/>


# 4) Example
- First one is the input track and the next three tracks are the top-3 recommendations:
<img src="pics/14.png" alt="drawing" width="400"/>

# 4) Group Members
[Mohamadreza Shariati](https://gitlab.com/Mrezashariati)<br>
[Farzaneh Rasti](https://gitlab.com/farzaneh_rst)<br> 
[Maryam Valipour](https://github.com/maryam-v) <br>
[Behnam Vakili](https://gitlab.com/behnam.vr)<br>
