"""Generate tests/fixtures/mini_corpus/articles.jsonl.

Provenance: every article below is original, hand-written prose (not copied
from Wikipedia or any other source). Facts are limited to widely cited,
checkable figures. doc_ids are derived deterministically from source_uri.

Run from the repo root:
    python3 tests/fixtures/mini_corpus/make_articles.py
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

INGESTED_AT = "2023-11-01T00:00:00Z"
OUT = Path(__file__).resolve().parent / "articles.jsonl"

# (title, paragraph 1, paragraph 2)
ARTICLES: list[tuple[str, str, str]] = [
    (
        "Mount Everest",
        "Mount Everest is the highest mountain above sea level, rising to 8,849 metres according to a joint survey announced by China and Nepal in 2020. It sits in the Mahalangur range of the Himalayas on the border between Nepal and the Tibet region of China, where it is known locally as Sagarmatha and Chomolungma.",
        "The British named the peak after George Everest, a former Surveyor General of India. Edmund Hillary of New Zealand and Tenzing Norgay, a Sherpa mountaineer, made the first confirmed ascent on 29 May 1953. Their expedition was led by John Hunt.",
    ),
    (
        "Marie Curie",
        "Marie Curie was a physicist and chemist born Maria Sklodowska in Warsaw in 1867, who later did most of her research in Paris. Working with her husband Pierre Curie, she announced the discovery of two new elements, polonium and radium, in 1898.",
        "She shared the 1903 Nobel Prize in Physics with Pierre Curie and Henri Becquerel for work on radioactivity. In 1911 she received the Nobel Prize in Chemistry on her own, making her the first person to win Nobel Prizes in two different sciences. She died in 1934 from aplastic anaemia linked to long exposure to radiation.",
    ),
    (
        "Danube",
        "The Danube is the second-longest river in Europe after the Volga, running for roughly 2,850 kilometres. It rises in the Black Forest of southwestern Germany and ends in a wide delta on the Black Sea shared by Romania and Ukraine.",
        "Along its course the river passes through or borders ten countries, more than any other river in the world. Four national capitals stand on its banks: Vienna, Bratislava, Budapest and Belgrade. The Blue Danube waltz by Johann Strauss II, which premiered in 1867, made the river famous in music.",
    ),
    (
        "Jupiter",
        "Jupiter is the fifth planet from the Sun and the largest in the solar system, with a mass more than twice that of all the other planets combined. A day on Jupiter lasts just under 10 hours, while one orbit around the Sun takes about 11.9 Earth years.",
        "In 1610 Galileo Galilei observed its four largest moons, now called Io, Europa, Ganymede and Callisto. Ganymede is the largest moon in the solar system and is wider than the planet Mercury. The NASA spacecraft Juno entered orbit around Jupiter in July 2016.",
    ),
    (
        "Eiffel Tower",
        "The Eiffel Tower stands on the Champ de Mars in Paris and was completed in 1889 as the entrance arch to the Exposition Universelle, held to mark the centenary of the French Revolution. It was built by the engineering company of Gustave Eiffel, based on an initial design by the engineers Maurice Koechlin and Emile Nouguier.",
        "The wrought iron structure was the tallest human-made structure in the world until the Chrysler Building in New York surpassed it in 1930. With its antennas, the tower now reaches about 330 metres. It was originally intended to stand for only 20 years.",
    ),
    (
        "Penicillin",
        "Penicillin was the first widely used antibiotic and is produced by moulds of the genus Penicillium. The Scottish scientist Alexander Fleming noticed its bacteria-killing effect in 1928 at St Mary's Hospital in London after a mould contaminated one of his culture plates.",
        "Around 1940 Howard Florey and Ernst Chain led a team at the University of Oxford that purified the drug and showed it could treat infections. Large-scale production in the United States and Britain made penicillin available to Allied troops during the Second World War. Fleming, Florey and Chain shared the 1945 Nobel Prize in Physiology or Medicine.",
    ),
    (
        "Lake Baikal",
        "Lake Baikal in southern Siberia, Russia, is the deepest lake on Earth, reaching a maximum depth of about 1,642 metres. Scientists estimate it to be around 25 million years old, which makes it the oldest lake in the world.",
        "Baikal holds roughly a fifth of the planet's unfrozen fresh surface water. More than 300 rivers flow into it, with the Selenga being the largest, but only the Angara River drains out. The lake is home to the Baikal seal, one of the few freshwater seal species, and was listed as a UNESCO World Heritage Site in 1996.",
    ),
    (
        "Magna Carta",
        "Magna Carta is a charter of rights agreed by King John of England at Runnymede, near Windsor, on 15 June 1215. It was drawn up to make peace between the unpopular king and a group of rebel barons, with Archbishop of Canterbury Stephen Langton acting as a mediator.",
        "Pope Innocent III annulled the charter only a few months later, and the dispute led to the First Barons' War. Revised versions were issued in 1216, 1217 and 1225 under John's son Henry III. Four original copies of the 1215 text survive, two at the British Library and one each at Lincoln and Salisbury cathedrals.",
    ),
    (
        "Machu Picchu",
        "Machu Picchu is a 15th-century Inca citadel on a mountain ridge in the Andes of southern Peru, about 2,430 metres above sea level. It overlooks the Urubamba River valley northwest of Cusco and is thought to have been built as an estate for the Inca emperor Pachacuti.",
        "The site was largely unknown to the outside world until the American explorer Hiram Bingham publicised it in 1911. UNESCO declared it a World Heritage Site in 1983. Its dry-stone walls, cut so precisely that no mortar was needed, are among the finest examples of Inca construction.",
    ),
    (
        "Oxygen",
        "Oxygen is a chemical element with the symbol O and atomic number 8. It makes up about 21 percent of Earth's atmosphere by volume and is the most abundant element in the planet's crust by mass.",
        "The Swedish chemist Carl Wilhelm Scheele produced oxygen in the early 1770s, and the English chemist Joseph Priestley independently reported it in 1774. The French chemist Antoine Lavoisier gave the element its name and explained its role in combustion. Oxygen becomes a pale blue liquid when cooled below about minus 183 degrees Celsius.",
    ),
    (
        "Isaac Newton",
        "Isaac Newton was an English mathematician and physicist born at Woolsthorpe Manor in Lincolnshire. His book Philosophiae Naturalis Principia Mathematica, published in 1687, set out the three laws of motion and the law of universal gravitation.",
        "Newton also developed calculus, independently of the German mathematician Gottfried Wilhelm Leibniz, and showed with prisms that white light is a mixture of colours. He held the Lucasian Professorship of Mathematics at Cambridge from 1669 and served as President of the Royal Society from 1703 until his death in 1727. He was also Master of the Royal Mint.",
    ),
    (
        "Charles Darwin",
        "Charles Darwin was an English naturalist born in Shrewsbury on 12 February 1809. From 1831 to 1836 he sailed around the world aboard HMS Beagle, visiting the Galapagos Islands in 1835.",
        "His observations on the voyage helped him develop the theory of evolution by natural selection. He published On the Origin of Species on 24 November 1859, after Alfred Russel Wallace had independently reached similar conclusions. Darwin died in 1882 and was buried in Westminster Abbey in London.",
    ),
    (
        "Saturn",
        "Saturn is the sixth planet from the Sun and the second largest in the solar system after Jupiter. Its average density is lower than that of water, and its famous rings are made mostly of chunks of water ice.",
        "The Dutch astronomer Christiaan Huygens discovered Titan, Saturn's largest moon, in 1655. Titan is the only moon known to have a thick atmosphere. The Cassini spacecraft orbited Saturn from 2004 until September 2017, when it was deliberately sent into the planet's atmosphere to end the mission.",
    ),
    (
        "Mars",
        "Mars is the fourth planet from the Sun and gets its reddish colour from iron oxide in its surface dust. A Martian solar day lasts about 24 hours and 40 minutes, only slightly longer than a day on Earth.",
        "The planet has two small moons, Phobos and Deimos, both discovered by the American astronomer Asaph Hall in 1877. Olympus Mons on Mars is the tallest known volcano in the solar system, rising roughly 22 kilometres high. NASA's Curiosity rover landed in Gale Crater in August 2012, and Perseverance landed in Jezero Crater in February 2021.",
    ),
    (
        "The Moon",
        "The Moon is Earth's only natural satellite, with a diameter of about 3,474 kilometres. It orbits at an average distance of about 384,400 kilometres and always shows the same face to Earth because its rotation is tidally locked.",
        "The time from one new moon to the next is about 29.5 days. Neil Armstrong and Buzz Aldrin landed on its surface on 20 July 1969 during the Apollo 11 mission, becoming the first people to walk on another world. Twelve astronauts in total walked on the Moon between 1969 and 1972.",
    ),
    (
        "Hydrogen",
        "Hydrogen is the lightest chemical element, with the symbol H and atomic number 1. It is the most abundant element in the universe, making up roughly three quarters of all ordinary matter by mass.",
        "The English scientist Henry Cavendish identified hydrogen as a distinct substance in 1766, calling it inflammable air. Antoine Lavoisier later named it from Greek words meaning water-former, because burning it produces water. Hydrogen gas boils at about minus 253 degrees Celsius, and in stars like the Sun it is fused into helium to release energy.",
    ),
    (
        "Gold",
        "Gold is a dense, soft, yellow metal with the chemical symbol Au, taken from the Latin word aurum, and atomic number 79. It melts at about 1,064 degrees Celsius and has a density of roughly 19.3 grams per cubic centimetre.",
        "Gold is so malleable that a single gram can be beaten into a sheet covering about one square metre. The California Gold Rush began after James Marshall found gold at Sutter's Mill in January 1848. In 1886 the discovery of gold on the Witwatersrand in South Africa led to the founding of Johannesburg.",
    ),
    (
        "Albert Einstein",
        "Albert Einstein was a theoretical physicist born in Ulm, Germany, on 14 March 1879. In 1905, while working at the patent office in Bern, he published papers on the photoelectric effect, Brownian motion, special relativity and the equivalence of mass and energy.",
        "He completed his general theory of relativity in 1915. Einstein received the 1921 Nobel Prize in Physics for explaining the photoelectric effect rather than for relativity. He moved to the United States in 1933, worked at the Institute for Advanced Study in Princeton, and died there in 1955.",
    ),
    (
        "Galileo Galilei",
        "Galileo Galilei was an Italian astronomer and physicist born in Pisa in 1564. After hearing about the newly invented telescope, he built improved versions and in 1610 published Sidereus Nuncius, describing mountains on the Moon and four moons orbiting Jupiter.",
        "He also observed the phases of Venus, which supported the Copernican idea that the planets orbit the Sun. His 1632 book Dialogue Concerning the Two Chief World Systems led to a trial by the Roman Inquisition in 1633. He spent his remaining years under house arrest near Florence and died in 1642.",
    ),
    (
        "Louis Pasteur",
        "Louis Pasteur was a French chemist and microbiologist born in Dole in 1822. His experiments helped establish the germ theory of disease, and the process of heating liquids such as wine and milk to kill microbes is called pasteurisation after him.",
        "In July 1885 he successfully treated a boy named Joseph Meister, who had been bitten by a rabid dog, using an experimental rabies vaccine. The Pasteur Institute in Paris was founded in 1887 to continue this work. Pasteur died in 1895.",
    ),
    (
        "Hubble Space Telescope",
        "The Hubble Space Telescope was launched on 24 April 1990 aboard the Space Shuttle Discovery. It is a joint project of NASA and the European Space Agency and is named after the American astronomer Edwin Hubble.",
        "Soon after launch, engineers found that its 2.4-metre primary mirror had been ground to a slightly wrong shape, blurring its images. Astronauts on the Space Shuttle Endeavour installed corrective optics during a servicing mission in December 1993. Hubble observations have since helped astronomers measure the expansion rate of the universe.",
    ),
    (
        "DNA Double Helix",
        "The double helix structure of DNA was described by James Watson and Francis Crick in a short paper in the journal Nature in April 1953. Their model relied heavily on X-ray diffraction images made at King's College London, including a famous image known as Photo 51 taken in Rosalind Franklin's laboratory.",
        "In the model, two strands wind around each other and are held together by base pairs, with adenine pairing with thymine and guanine with cytosine. Watson, Crick and Maurice Wilkins shared the 1962 Nobel Prize in Physiology or Medicine. Franklin had died in 1958 and could not be considered, since the prize is not awarded posthumously.",
    ),
    (
        "Amazon River",
        "The Amazon River flows across northern South America from the Andes of Peru to the Atlantic Ocean in Brazil. It carries more water than any other river, with an average discharge of about 209,000 cubic metres per second, roughly a fifth of all river water reaching the oceans.",
        "Its length is usually given as about 6,400 kilometres, making it either the first or second longest river depending on how its source is defined. The river basin covers roughly 7 million square kilometres. The Spanish explorer Francisco de Orellana completed the first European descent of the river in 1542.",
    ),
    (
        "Sahara Desert",
        "The Sahara is the largest hot desert in the world, covering about 9 million square kilometres of northern Africa. It stretches from the Atlantic Ocean in the west to the Red Sea in the east.",
        "Its highest point is Emi Koussi, a volcano in the Tibesti Mountains of Chad that rises to about 3,415 metres. Much of the desert is rocky plateau rather than sand dunes. Rock art in places such as Tassili n'Ajjer in Algeria shows that parts of the Sahara were much greener several thousand years ago.",
    ),
    (
        "Nile",
        "The Nile flows north through northeastern Africa into the Mediterranean Sea and is about 6,650 kilometres long, traditionally ranked as the longest river in the world. Its two main branches are the White Nile, which drains the region around Lake Victoria, and the Blue Nile, which starts at Lake Tana in Ethiopia.",
        "The two branches meet at Khartoum, the capital of Sudan. In Egypt the Aswan High Dam, completed in 1970, created Lake Nasser and ended the annual summer flood. Ancient Egyptian farming depended on the fertile silt left behind by that flood.",
    ),
    (
        "Mount Kilimanjaro",
        "Mount Kilimanjaro in northeastern Tanzania is the highest mountain in Africa, with its summit at Uhuru Peak reaching 5,895 metres. It is a dormant volcano made up of three cones named Kibo, Mawenzi and Shira.",
        "The German geographer Hans Meyer and the Austrian climber Ludwig Purtscheller made the first recorded ascent in 1889. Kilimanjaro National Park was listed as a UNESCO World Heritage Site in 1987. The glaciers near the summit have shrunk sharply over the past century.",
    ),
    (
        "Great Barrier Reef",
        "The Great Barrier Reef lies in the Coral Sea off the coast of Queensland, Australia, and is the largest coral reef system in the world. It stretches for more than 2,300 kilometres and is made up of about 2,900 individual reefs and some 900 islands.",
        "The Great Barrier Reef Marine Park was established in 1975 to protect the area. UNESCO listed the reef as a World Heritage Site in 1981. In recent decades it has suffered several mass coral bleaching events linked to unusually warm sea temperatures.",
    ),
    (
        "Pacific Ocean",
        "The Pacific Ocean is the largest and deepest of Earth's oceans, covering about 165 million square kilometres, roughly a third of the planet's surface. It stretches from the Arctic in the north to the Southern Ocean in the south, between Asia and Australia on one side and the Americas on the other.",
        "The explorer Ferdinand Magellan gave the ocean its name, meaning peaceful, after crossing calm waters in 1520. The Mariana Trench in the western Pacific contains the deepest known point on Earth. The ocean's edges are lined by the Ring of Fire, a belt of frequent earthquakes and volcanoes.",
    ),
    (
        "Dead Sea",
        "The Dead Sea is a salt lake in the Jordan Rift Valley, fed mainly by the Jordan River and with no outlet. Its shore lies about 430 metres below sea level, the lowest land elevation on Earth.",
        "The water is about 34 percent salt, nearly ten times saltier than the ocean, so swimmers float easily and few organisms other than microbes can survive. Since the 1970s the water level has been falling by around a metre a year. The Dead Sea Scrolls were discovered from 1947 onward in caves at nearby Qumran.",
    ),
    (
        "Mariana Trench",
        "The Mariana Trench is a deep crescent-shaped trench in the floor of the western Pacific Ocean, east of the Mariana Islands. Its deepest known point, the Challenger Deep, lies nearly 11,000 metres below the sea surface.",
        "On 23 January 1960, Jacques Piccard and Don Walsh reached the bottom of the Challenger Deep in the bathyscaphe Trieste. In 2012 the film director James Cameron made the first solo dive to the bottom in the submersible Deepsea Challenger. The trench forms where the Pacific Plate is pushed beneath the Mariana Plate.",
    ),
    (
        "Victoria Falls",
        "Victoria Falls is a waterfall on the Zambezi River on the border between Zambia and Zimbabwe. It is about 1,708 metres wide and up to 108 metres high, forming one of the largest curtains of falling water in the world.",
        "The local name, Mosi-oa-Tunya, means the smoke that thunders, describing the spray that can be seen from many kilometres away. The Scottish missionary David Livingstone saw the falls in 1855 and named them after Queen Victoria. The falls were listed as a UNESCO World Heritage Site in 1989.",
    ),
    (
        "Lake Superior",
        "Lake Superior is the largest of the Great Lakes of North America and the largest freshwater lake in the world by surface area, at about 82,100 square kilometres. It is bordered by the Canadian province of Ontario and the US states of Minnesota, Wisconsin and Michigan.",
        "The lake reaches a maximum depth of about 406 metres and drains into Lake Huron through the St Marys River. The freighter SS Edmund Fitzgerald sank in a storm on Lake Superior in November 1975 with all 29 crew members. The loss was later commemorated in a 1976 song by Gordon Lightfoot.",
    ),
    (
        "Iceland",
        "Iceland is a Nordic island country in the North Atlantic Ocean with an area of about 103,000 square kilometres. Its capital, Reykjavik, is the northernmost capital of a sovereign state.",
        "The Icelandic parliament, the Althing, was founded in 930 and is one of the oldest parliaments in the world. Iceland became a fully independent republic on 17 June 1944, ending its union with Denmark. The island sits on the Mid-Atlantic Ridge, and the 2010 eruption of Eyjafjallajokull closed much of European airspace for several days.",
    ),
    (
        "Antarctica",
        "Antarctica is the southernmost continent, covering about 14 million square kilometres, almost all of it under a thick ice sheet. This ice holds around 70 percent of the world's fresh water.",
        "The lowest natural air temperature ever recorded by a ground station, minus 89.2 degrees Celsius, was measured at the Soviet Vostok Station on 21 July 1983. A Norwegian expedition led by Roald Amundsen became the first to reach the South Pole on 14 December 1911. The Antarctic Treaty, signed by 12 countries in 1959, set the continent aside for peaceful scientific research.",
    ),
    (
        "Mount Fuji",
        "Mount Fuji is the highest mountain in Japan, rising to 3,776 metres on the island of Honshu between Shizuoka and Yamanashi prefectures. It is an active stratovolcano whose most recent eruption took place in 1707 and 1708.",
        "The nearly symmetrical cone appears in many works of Japanese art, including the woodblock print series Thirty-six Views of Mount Fuji by Katsushika Hokusai. UNESCO added Mount Fuji to its World Heritage List as a cultural site in 2013. The climbing season on the main trails usually runs during July and August.",
    ),
    # ---- partner (7) ----
    (
        "Great Wall of China",
        "The Great Wall of China is a series of fortifications built across northern China over many centuries to guard against raids from the steppe. After unifying China in 221 BC, the first emperor Qin Shi Huang ordered earlier walls to be joined together.",
        "Most of the brick and stone sections seen by visitors today, such as Badaling near Beijing, were built during the Ming dynasty, which ruled from 1368 to 1644. A Chinese government survey published in 2012 measured all branches of the wall at about 21,196 kilometres. The wall was listed as a UNESCO World Heritage Site in 1987.",
    ),
    (
        "Colosseum",
        "The Colosseum in the centre of Rome, originally known as the Flavian Amphitheatre, is the largest amphitheatre ever built in the Roman world. Construction began under Emperor Vespasian around AD 72 and was completed under his son Titus in AD 80.",
        "The arena is estimated to have held between 50,000 and 80,000 spectators, who came to watch gladiator fights and animal hunts. Titus marked its opening with games that lasted 100 days. It was built mainly from travertine limestone, concrete and tuff.",
    ),
    (
        "Apollo 11",
        "Apollo 11 was the NASA mission that first landed humans on the Moon. It launched from Kennedy Space Center in Florida on 16 July 1969 on a Saturn V rocket, carrying Neil Armstrong, Buzz Aldrin and Michael Collins.",
        "The lunar module Eagle touched down in the Sea of Tranquility on 20 July 1969 while Collins stayed in orbit aboard the command module Columbia. Armstrong and Aldrin spent about two and a half hours walking outside and collected about 21.5 kilograms of lunar samples. The crew splashed down safely in the Pacific Ocean on 24 July 1969.",
    ),
    (
        "Statue of Liberty",
        "The Statue of Liberty is a copper statue on Liberty Island in New York Harbor, given to the United States by the people of France. It was designed by the French sculptor Frederic Auguste Bartholdi, while its internal iron framework was engineered by Gustave Eiffel.",
        "The statue was dedicated on 28 October 1886. From the ground to the tip of the torch it stands about 93 metres tall, including its pedestal. Its green colour comes from a natural patina that formed on the copper over several decades. It became a UNESCO World Heritage Site in 1984.",
    ),
    (
        "Printing Press",
        "Johannes Gutenberg, a goldsmith from Mainz in Germany, developed a printing press with movable metal type around 1440. His system combined a hand mould for casting letters, an oil-based ink and a press adapted from those used for wine and paper.",
        "The Gutenberg Bible, completed in Mainz around 1455, was the first major book printed with this method in Europe. Fewer than 50 copies survive today, many of them incomplete. William Caxton set up the first printing press in England at Westminster in 1476.",
    ),
    (
        "Taj Mahal",
        "The Taj Mahal is a white marble mausoleum on the south bank of the Yamuna River in Agra, India. The Mughal emperor Shah Jahan commissioned it in 1632 in memory of his wife Mumtaz Mahal, who had died in 1631.",
        "The main tomb was finished around 1643, and the surrounding gardens and buildings were completed about ten years later. Around 20,000 artisans and workers are thought to have taken part in the construction. UNESCO listed the Taj Mahal as a World Heritage Site in 1983.",
    ),
    (
        "Panama Canal",
        "The Panama Canal is an artificial waterway about 82 kilometres long that links the Atlantic and Pacific oceans across the Isthmus of Panama. A French company led by Ferdinand de Lesseps began work in 1881 but abandoned the effort after disease and financial collapse.",
        "The United States took over the project in 1904 and opened the canal on 15 August 1914. Its locks lift ships about 26 metres up to Gatun Lake, an artificial lake created by damming the Chagres River. Panama took full control of the canal on 31 December 1999, and a larger set of locks opened in 2016.",
    ),
    # ---- web (5) ----
    (
        "Titanic",
        "RMS Titanic was a British passenger liner built by Harland and Wolff in Belfast for the White Star Line. On its maiden voyage from Southampton to New York, it struck an iceberg late on 14 April 1912 and sank early the next morning.",
        "More than 1,500 of the roughly 2,200 people on board died, partly because the ship carried too few lifeboats for everyone. The ship was commanded by Captain Edward Smith, who went down with it. A team led by Robert Ballard and Jean-Louis Michel found the wreck in the North Atlantic in 1985.",
    ),
    (
        "Stonehenge",
        "Stonehenge is a prehistoric monument on Salisbury Plain in Wiltshire, England, built in several stages between about 3000 BC and 2000 BC. Its large sarsen stones were brought from the area around West Woods, roughly 25 kilometres to the north.",
        "The smaller bluestones were transported about 240 kilometres from the Preseli Hills in west Wales. The monument is aligned with the sunrise at the summer solstice and the sunset at the winter solstice. Together with Avebury, it was inscribed as a UNESCO World Heritage Site in 1986.",
    ),
    (
        "Rosetta Stone",
        "The Rosetta Stone is a granodiorite slab carrying a decree issued in 196 BC on behalf of the Egyptian king Ptolemy V. French soldiers found it in 1799 near the town of Rashid, known to Europeans as Rosetta, in the Nile Delta.",
        "The decree is written three times, in Egyptian hieroglyphs, Demotic script and ancient Greek. Because the Greek text could be read, scholars used the stone as a key to Egyptian writing, and Jean-Francois Champollion announced his decipherment of hieroglyphs in 1822. The stone has been displayed at the British Museum in London since 1802.",
    ),
    (
        "Suez Canal",
        "The Suez Canal is a sea-level waterway in Egypt that connects the Mediterranean Sea at Port Said with the Red Sea at Suez, a distance of about 193 kilometres. It was built by a company organised by the French diplomat Ferdinand de Lesseps and opened on 17 November 1869.",
        "Because it has no locks, ships travel between the two seas without being raised or lowered. Egyptian President Gamal Abdel Nasser nationalised the canal in 1956, triggering the Suez Crisis. In March 2021 the container ship Ever Given ran aground in the canal and blocked it for six days.",
    ),
    (
        "Golden Gate Bridge",
        "The Golden Gate Bridge is a suspension bridge spanning the Golden Gate strait at the entrance to San Francisco Bay in California. It opened to pedestrians on 27 May 1937 and to vehicles the following day.",
        "Its main span of 1,280 metres was the longest of any suspension bridge in the world until 1964. Joseph Strauss was the chief engineer, and the architect Irving Morrow chose its distinctive International Orange colour. The two towers rise about 227 metres above the water.",
    ),
    # ---- user_upload (3) ----
    (
        "Battle of Hastings",
        "The Battle of Hastings was fought on 14 October 1066 between the Norman army of William, Duke of Normandy, and the English army of King Harold II. It took place about 11 kilometres northwest of Hastings, at a site where the town of Battle in East Sussex now stands.",
        "Only weeks earlier, on 25 September, Harold had defeated a Norwegian invasion under Harald Hardrada at Stamford Bridge. Harold was killed at Hastings, and William was crowned king in Westminster Abbey on Christmas Day 1066. The events are illustrated in the Bayeux Tapestry, an embroidered cloth about 70 metres long.",
    ),
    (
        "Wright Brothers",
        "Wilbur and Orville Wright were American brothers who ran a bicycle business in Dayton, Ohio, before turning to aviation. On 17 December 1903, at Kill Devil Hills near Kitty Hawk, North Carolina, they made the first controlled, sustained flights of a powered, heavier-than-air aircraft.",
        "Orville piloted the first flight, which lasted 12 seconds and covered about 37 metres. Later that day Wilbur flew about 260 metres in 59 seconds, the longest of the four flights. The original 1903 Wright Flyer is displayed at the Smithsonian National Air and Space Museum in Washington.",
    ),
    (
        "Leaning Tower of Pisa",
        "The Leaning Tower of Pisa is the freestanding bell tower of Pisa Cathedral in the Tuscany region of Italy. Construction began in 1173, and the tower started to tilt during building because its foundations rested on soft ground.",
        "Work stopped for long periods, and the bell chamber was not completed until 1372. Engineers carried out stabilisation work between 1990 and 2001 that reduced the lean to about 4 degrees. The tower stands about 56 metres tall and forms part of the Piazza del Duomo, a UNESCO World Heritage Site since 1987.",
    ),
]

TIERS = ["curated"] * 35 + ["partner"] * 7 + ["web"] * 5 + ["user_upload"] * 3

_SENTENCE_END = re.compile(r"\.(?=\s|$)")


def slugify(title: str) -> str:
    slug = title.lower().replace(" ", "-")
    assert re.fullmatch(r"[a-z0-9-]+", slug), slug
    return slug


def build() -> list[dict]:
    assert len(ARTICLES) == 50, len(ARTICLES)
    assert len({t for t, _, _ in ARTICLES}) == 50, "titles must be distinct"
    records = []
    for (title, p1, p2), tier in zip(ARTICLES, TIERS, strict=True):
        raw = f"{p1}\n\n{p2}"
        assert raw.isascii(), title
        words = len(raw.split())
        sentences = len(_SENTENCE_END.findall(raw))
        assert 70 <= words <= 130, (title, words)
        assert 4 <= sentences <= 6, (title, sentences)
        source_uri = f"fixture://mini_corpus/{slugify(title)}"
        records.append(
            {
                "doc_id": hashlib.sha1(source_uri.encode()).hexdigest()[:16],
                "title": title,
                "source_uri": source_uri,
                "fmt": "text",
                "raw": raw,
                "source_tier": tier,
                "ingested_at": INGESTED_AT,
            }
        )
    return records


def main() -> None:
    records = build()
    with OUT.open("w", encoding="ascii", newline="\n") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=True) + "\n")
    print(f"wrote {len(records)} records to {OUT}")


if __name__ == "__main__":
    main()
