# Normalår: 60 % femårs-NPV och 40 % kvalitet

Användaren valde den 29 september 2026 att låta **NPV för kassaflöden under år
1–5 väga 60 %**, och kvalitet väga **40 %**, i de sparade normalårslistornas
sortering. Femårsmåttet räknar inget terminalvärde. Policyn heter
`normal-quality-watch-npv5-60-40-v2`.

De befintliga kvalitetskraven, undantagsåren 2020–2023 och den strikta rabattlistans
gränser behålls. Äldre normalårsvärderingar med tio års kassaflöden och olika
terminalandelar visas fortfarande separat. Personliga värderingsutkast och
allårsmodellen påverkas inte.

## Fem års modellerade kassaflöden

Låt C vara medianen av leverantörens fria kassaflöde under de fem valda normalåren.
Anta ett oförändrat nominellt kassaflöde C under vart och ett av de kommande fem
åren. Diskonteringsräntan är fortsatt 10 %.

`PV5 = C/1,10 + C/1,10² + C/1,10³ + C/1,10⁴ + C/1,10⁵`

`NPV5 = PV5 − sparat börsvärde`

`NPV5/pris = 100 × (PV5 / sparat börsvärde − 1)`

PV5 motsvarar **3,7907867694 × C**. Kassaflöden från år 6 och framåt bidrar med
noll till just detta mått. Kassa och skuld läggs inte till en gång till.
Leverantörens fria kassaflöde är fortsatt ett underlag som behöver stämmas av
mot ägarkassaflöde, investeringar, leasing och finansiering.

Ett negativt NPV5 betyder att dessa fem diskonterade kassaflöden inte täcker hela
det sparade börsvärdet. Måttet ger inget besked om utfallet av en framtida
försäljning av aktien. Jämförelsen kräver rätt valuta, en giltig aktie- och
kursgrund samt ett sparat pris som är högst 550 dagar gammalt vid snapshoten.
Femårs-PV kan visas även när en giltig prisjämförelse saknas.

## Fast jämförelsegrupp och tydliga vikter

Samma fasta normalårsgrupp används som i
[den tidigare rankningen](normal-years-ranking-2026-09-29.md). Sökning, land,
bransch, bevakningslista och synliga värderingsgränser påverkar inte gruppen som
percentilerna beräknas mot.

Den sparade datan ger **248 kvalitetsnoteringar**. **180** har fullständiga
underlag för poängen. **68** saknar giltig prisjämförelse och visas sist utan
sammanvägd poäng. Saknade uppgifter ersätts inte med noll och vikterna fördelas
inte om. Den ursprungliga strikta rabattlistan har fortfarande **0 träffar**.

Gruppen är viktad per notering. Flera noteringar av ett företag kan därför
påverka percentilerna. Bolagsnamn används inte för att gissa gemensam identitet,
och ISIN behandlas inte som ett säkert gemensamt bolags-ID.

| Del | Mått | Riktning | Högsta bidrag till totalen |
| --- | --- | --- | ---: |
| Femårs-NPV | NPV för år 1–5 / sparat börsvärde, utan terminalvärde | Högre | 60 poäng |
| Kapitalavkastning | Median EBIT / (eget kapital + nettoskuld) under normalåren | Högre | 10 poäng |
| Marginaluthållighet | Lägsta EBIT-marginal under normalåren | Högre | 10 poäng |
| Kassaflödestillväxt | Rörelsekassaflödets CAGR mellan normalårens ändpunkter | Högre | 10 poäng |
| Balansräkning | Senaste nettoskuld / EBITDA, där nettokassa behandlas som noll | Lägre | 10 poäng |

Varje mått får en percentilpoäng 0–100. Lika värden delar medelpositionen:

`percentil = 100 × (antal strikt sämre + (antal lika − 1) / 2) / (N − 1)`

`kvalitet = medelvärdet av de fyra kvalitetspercentilerna`

`totalpoäng = 0,60 × femårs-NPV-percentil + 0,40 × kvalitet`

Kvalitetsmedelvärdet percentilrankas inte igen. Nettokassa behåller sitt
ursprungliga negativa värde i tabellen, men skuldpercentilen använder
`max(nettoskuld/EBITDA, 0)`. Ett extremt högt bokföringsmässigt avkastningstal
kan därmed bidra med högst 10 poäng till totalen.

För att matematiskt lika poäng verkligen ska bli lika används dubblerade
rangpositioner som heltal före divisionen. Om `r2 = 2 × antal strikt sämre +
antal lika − 1` gäller:

`totalpoäng = 100 × (6 × femårs-NPV-r2 + summan av fyra kvalitets-r2) / (20 × (N − 1))`

En grupp med en enda notering får 50 poäng. Sorteringen är totalpoäng fallande,
därefter det råa femårs-NPV/pris-talet fallande, och slutligen namn och stabilt
noterings-ID. Saknad totalpoäng ligger sist.

## Vad ändringen faktiskt gör i den sparade datan

Eftersom samtliga bolag använder samma konstanta kassaflödesantagande och samma
diskonteringsränta är femårs-NPV/pris en positiv linjär omräkning av det tidigare
måttet med tio år och halvt terminalvärde. Därför är **alla 180
värderingspercentiler oförändrade**. Kvalitetspercentilerna är också oförändrade.
Den nya ordningen kommer från ändringen av vikterna till 60/40: **173
rankningspositioner ändras**.

Ingen av de 180 prisjämförbara noteringarna har positivt eller noll femårs-NPV i
denna snapshot. Det högsta NPV5/pris-talet är ungefär **−47,60 %**, och det
lägsta ungefär **−99,00 %**. En hög relativ poäng betyder alltså att en notering
står sig väl i jämförelsen; den garanterar inte att fem års kassaflöden täcker
priset eller att bolaget är undervärderat.

Den ekonomiska tolkningen behöver även ta hänsyn till att de fem normalåren ofta
är 2017–2019 samt 2024–2025. Medianen kan representera en äldre verksamhetsskala.
Undantagsårens rådata är kvar för granskning av exempelvis förluster, utspädning
och likviditetsproblem. Rankningen är ett användarvalt analysstöd och anger inga
förväntade avkastningar eller sannolikheter.

## Reproducerbar kontroll

- Forskningssnapshot: 2026-09-13. Leverantörssnapshot: 2026-08-10.
- Gauge SHA-256: `4b7d4298e0f1b5e74c2cad03e53900f746287829ce6113c45851ff75cfe65645`.
- Oberoende referensidentitet: `5afc66b853bbe44572800bf5ebe6b54a83c736a6a6ed4de6b773fe1990e1b829`.
- Sparad referens: `desktop/tests/fixtures/normal-year-five-year-ranking.json`,
  SHA-256 `912f56e957f8458c4224a1423d0b7530abb5ef5ce07dd99e901006ee36a4b2c9`.
- Versionshanterade kontrollskript finns i `desktop/scripts/verification/`.
  Python räknar om underlaget och kräver byteidentisk överensstämmelse med
  referensen. Node jämför därefter med appens rankning, cellvärden, parser och
  sortering. Äldre tioårsmått räknas också om oberoende i Python.
- Kontrollen kräver de ursprungliga lokala gauge- och KPI-filerna under
  `desktop/public/data/`, Python 3 och projektets installerade Node-beroenden.
  Varje datafil kontrolleras mot manifestets storlek och SHA-256. De stora
  datafilerna distribueras separat från Git.

Kör från `desktop/`, med en ny resultatmapp:

```bash
ranking_audit_output="test-results/normal-year-verification-$(date -u +%Y%m%dT%H%M%SZ)"
python3 scripts/verification/build-normal-year-ranking.py "$ranking_audit_output"
node scripts/verification/check-normal-year-ranking.mjs "$ranking_audit_output"
node scripts/verification/check-normal-year-model.mjs "$ranking_audit_output"
```

Resultaten, inklusive `expected-ranking.json`, `legacy-metrics.json` och två
kontrollkvitton, skrivs bara under den nya ignorerade resultatmappen. Sparade
testfixturer och tidigare releasekvitton skrivs inte över. Vid överlämningen
kontrollerades även samtliga **4 960** omräknade äldre mätvärden mot det lokalt
bevarade ursprungliga normalårsunderlaget; de stämde överens.

Den tidigare 50/50-beräkningen och dess testfixtur ligger kvar separat. Dess
historiska generator under `test-results/` är ett lokalt granskningsunderlag;
de versionshanterade kommandona ovan reproducerar den aktuella 60/40-policyn.
