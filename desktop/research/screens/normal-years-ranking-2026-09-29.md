# Normalår: 50 % rabatt och 50 % kvalitet

Användaren valde den 29 september 2026 att ge rabatt och kvalitet lika stor vikt i
sorteringen av de sparade normalårslistorna. Detta är en transparent rangordning
för vidare bolagsanalys. Poängen är inte en förväntad avkastning, sannolikhet eller
ny beräkning av verkligt värde.

## Referensgruppen ligger fast

Utgå från samtliga noteringar som uppfyller de befintliga villkoren i
**Kvalitetsbolag – normalår, prisbevakning**, innan sökning, land, bransch,
bevakningslista eller värderingsgränser begränsar det som visas. Alla fyra
kvalitetsmått och en giltig jämförelse med sparat aktiepris måste finnas.

Den frysta datan ger **248 kvalitetsnoteringar**. **180** har kompletta underlag
för sammanvägningen och utgör referensgruppen. De övriga **68** får ingen
sammanvägd poäng och visas sist; saknade uppgifter ersätts inte med noll och
vikterna fördelas inte om. Den striktare rabattlistans ursprungliga gränser
kvarstår och ger fortfarande **0 träffar**. En hög relativ poäng innebär alltså
inte i sig att ett bolag är undervärderat.

Referensgruppen är viktad per **notering**, inte per unikt bolag. Flera noteringar
av samma företag kan påverka fördelningen. Det finns inte ett tillräckligt säkert
gemensamt bolags-ID för att slå samman alla sådana poster. Namn används inte för
att gissa bolagsidentitet, och ISIN behandlas inte som ett bolags-ID.

Perioderna och beräkningarna följer den tidigare
[normalårspolicyn](normal-years-quality-value-2026-09-28.md). Rapporter som slutar
2020–2023 undantas, fem andra jämförbara årsrapporter används och CAGR räknas över
verklig förfluten tid. Skuldmåttet är en separat senasteobservation; det är inte
ett genomsnitt där undantagsåren har tagits bort.

## Poängen går att räkna om

Varje ingående mått omvandlas till en percentilpoäng mellan 0 och 100 inom samma
fasta grupp. För ett mått där högre är bättre gäller:

`percentil = 100 × (antal strikt sämre + (antal lika − 1) / 2) / (N − 1)`

Lika värden får medelpositionen för sina upptagna platser. Om gruppen endast
innehåller en notering blir poängen 50. Rangordningen bygger på hela observerade
tal, inte avrundade tabellvärden. I beräkningen summeras dubblerade rangpositioner
som heltal före den sista divisionen. Det bevarar matematiskt lika totalpoäng så
att små flyttalsfel inte kringgår den avsedda skiljesorteringen.

| Del | Ingående mått | Riktning | Högsta bidrag till totalen |
| --- | --- | --- | ---: |
| Rabatt | Normalårs-NPV / sparat pris, med 50 % av terminalvärdet | Högre | 50 poäng |
| Kapitalavkastning | Median EBIT / (eget kapital + nettoskuld) under normalåren | Högre | 12,5 poäng |
| Marginaluthållighet | Lägsta EBIT-marginal under normalåren | Högre | 12,5 poäng |
| Kassaflödestillväxt | Rörelsekassaflödets CAGR mellan normalårens ändpunkter | Högre | 12,5 poäng |
| Balansräkning | Senaste nettoskuld / EBITDA | Lägre | 12,5 poäng |

Vid skuldrankningen används `max(nettoskuld / EBITDA, 0)`. Alla nettokassor får
därmed samma behandling som noll nettoskuld; en allt större nettokassa ger inte
obegränsat högre poäng. Det ursprungliga signerade skuldmåttet visas fortfarande.

`kvalitet = (kapitalpercentil + marginalpercentil + kassaflödespercentil + skuldpercentil) / 4`

`totalpoäng = 0,50 × rabattpercentil + 0,50 × kvalitet`

Kvalitetsmedelvärdet percentilrankas inte en gång till. Vikterna är lika i poäng,
inte ett påstående om lika historisk förklaringskraft eller förväntad avkastning.
Ett extremt högt bokföringsmässigt avkastningstal kan bidra med högst 12,5 poäng
till totalen. Övriga befintliga kvalitetskrav, exempelvis avkastning på materiella
tillgångar, omsättningstillväxt och positivt kassaflöde, kvarstår som gränsvillkor.

Rabattdelen använder endast **en** värderingssignal. Under den fasta modellen
med konstant kassaflöde är full-, halv- och nollterminalfallen monotona
omräkningar av samma kassaflöde/pris-kvot. De visas som känslighetsfall men får
inte tre röster i rankningen. NPV/pris är ett överskott relativt priset, inte
rabatt uttryckt i procent av modellvärdet.

Sortera först på totalpoäng fallande, därefter halvterminal-NPV/pris fallande,
sedan namn och stabilt noterings-ID. Saknad totalpoäng ligger sist. Att ändra
sökning eller visningsfilter ändrar inte poängen för en kvarvarande notering.

## Underlag och kontroll

Referensgruppen är bunden till forskningsdatan från 2026-09-13 och
leverantörssnapshoten från 2026-08-10. Kurserna är sparade observationer och
uppdateras inte av denna sortering. Både rapport- och prisjämförelsen använder
de tidigare gränserna för aktualitet och valuta.

- Gauge SHA-256: `4b7d4298e0f1b5e74c2cad03e53900f746287829ce6113c45851ff75cfe65645`.
- Referensidentitet i den oberoende kontrollen: `de0d97cf39752827f68cccdfb1f4b7cba4974070bcbe74a397ec4cd44e6f0596`.
- Oberoende Python-beräkning: `desktop/test-results/normal-years-ranking-2026-09-29/build-ranking.py`.
- Förväntade poäng, komponenter, råvärden och ordning: `expected-ranking.json` i samma mapp.
- Jämförelse med TypeScript-modulen: `ranking-module-verification.json`.

Kapitalavkastningen är fortfarande en bokföringsproxy med kapital vid periodens
slut. Percentiler begränsar extrema tals vikt men gör inte ett felaktigt eller
missvisande underlag ekonomiskt korrekt. Prissättningsmakt, underhållsinvesteringar,
uthålliga ägarkassaflöden och riskerna under undantagsåren kräver fortsatt analys.
