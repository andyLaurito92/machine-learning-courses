# Perform the necessary imports
from sklearn.decomposition import NMF
from sklearn.preprocessing import Normalizer, MaxAbsScaler
from sklearn.pipeline import make_pipeline
import pandas as pd

# You are given a sparse array artists whose rows correspond to artists and whose columns correspond to users. The entries give the number of times each artist was listened to by each user.

# artist_names = ['Massive Attack', 'Sublime', 'Beastie Boys', 'Neil Young', 'Dead Kennedys', 'Orbital', 'Miles Davis', 'Leonard Cohen', 'Van Morrison', 'NOFX', 'Rancid', 'Lamb', 'Korn', 'Dropkick Murphys', 'Bob Dylan', 'Eminem', 'Nirvana', 'Van Halen', 'Damien Rice', 'Elvis Costello', 'Everclear', 'Jimi Hendrix', 'PJ Harvey', 'Red Hot Chili Peppers', 'Ryan Adams', 'Soundgarden', 'The White Stripes', 'Madonna', 'Eric Clapton', 'Bob Marley', 'Dr. Dre', 'The Flaming Lips', 'Tom Waits', 'Moby', 'Cypress Hill', 'Garbage', 'Fear Factory', '50 Cent', 'Ani DiFranco', 'Matchbox Twenty', 'The Police', 'Eagles', 'Phish', 'Stone Temple Pilots', 'Black Sabbath', 'Britney Spears', 'Fatboy Slim', 'System of a Down', 'Simon & Garfunkel', 'Snoop Dogg', 'Aimee Mann', 'Less Than Jake', 'Rammstein', 'Reel Big Fish', 'The Prodigy', 'Pantera', 'Foo Fighters', 'The Beatles', 'Incubus', 'Audioslave', 'Bright Eyes', 'Machine Head', 'AC/DC', 'Dire Straits', 'MotÃ¶rhead', 'Ramones', 'Slipknot', 'Me First and the Gimme Gimmes', 'Bruce Springsteen', 'Queens of the Stone Age', 'The Chemical Brothers', 'Bon Jovi', 'Goo Goo Dolls', 'Alice in Chains', 'Howard Shore', 'Barenaked Ladies', 'Anti-Flag', 'Nick Cave and the Bad Seeds', 'Static-X', 'Misfits', '2Pac', 'Sparta', 'Interpol', 'The Crystal Method', 'The Beach Boys', 'Goldfrapp', 'Bob Marley & the Wailers', 'Kylie Minogue', 'The Blood Brothers', 'Mirah', 'Ludacris', 'Snow Patrol', 'The Mars Volta', 'Yeah Yeah Yeahs', 'Iced Earth', 'Fiona Apple', 'Rilo Kiley', 'Rufus Wainwright', 'Flogging Molly', 'Hot Hot Heat', 'Dredg', 'Switchfoot', 'Tegan and Sara', 'Rage Against the Machine', 'Keane', 'Jet', 'Franz Ferdinand', 'The Postal Service', 'The Dresden Dolls', 'The Killers', 'Death From Above 1979']

##   (0, 2)	105.0
  # (0, 15)	165.0
  # (0, 20)	91.0
  # (0, 21)	98.0
  # (0, 29)	120.0
  # (0, 48)	236.0
  # (0, 70)	67.0
  # (0, 95)	77.0
  # (0, 96)	93.0
  # (0, 109)	98.0

# Create a MaxAbsScaler: scaler
scaler = MaxAbsScaler()

# Create an NMF model: nmf
nmf = NMF(n_components=20)

# Create a Normalizer: normalizer
normalizer = Normalizer()

# Create a pipeline: pipeline
pipeline = make_pipeline(scaler, nmf, normalizer)

# Apply fit_transform to artists: norm_features
norm_features = pipeline.fit_transform(artists)

# Create a DataFrame: df
df = pd.DataFrame(norm_features, index=artist_names)

# Select row of 'Bruce Springsteen': artist
artist = df.loc['Bruce Springsteen']

# Compute cosine similarities: similarities
similarities = df.dot(artist)

# Display those with highest cosine similarity
print(similarities.nlargest())
