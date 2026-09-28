# Exceptionella bolag – rimligt pris

Sparat den 28 september 2026 i den installerade Macro Atlas 0.22.1-profilen.
Öppna **Companies → Lists → Saved view → Exceptionella bolag – rimligt pris**.
Vyn sparar kolumner, exakta perioder, gränser och sortering för återanvändning.
Den portabla [filterposten](exceptional-fair-2026-09-28.json) innehåller definition,
daterade källidentiteter, träffar och verifiering.

## Syfte och värdering

Sök bolag med hög och uthållig historisk kapitalavkastning, god lönsamhet,
omsättningstillväxt, positiva kassaflöden och begränsad skuldsättning. Prisvillkoret
är **Starter Mid NPV / sparat pris ≥0 %**, med hela terminalvärdet inkluderat en
gång. Det betyder att priset högst motsvarar modellens centrala värde. Även lägre
priser tillåts. Det tidigare filtrets krav på 30 procents rabatt till ett värde
med halverad terminaldel används inte här.

Detta är en ny sparad vy. Den ändrar inte värderingsmodell, avkastningskrav,
köptak, positionsstorlek, användarens värderingsutkast eller det tidigare filtret.
Halverad terminaldel, ingen terminaldel och Low-scenariot visas som separata
känslighetsanalyser utan egna gränsvärden. De är inte sannolikheter eller
garanterade förlustgränser. Negativa utfall kan därför förekomma bland träffarna.

## Samtliga villkor

| Egenskap | Gräns |
|---|---|
| Bolagsurval | Rörelsedrivande bolag i senaste katalogen, fem jämförbara årsrapporter, ingen klassificeringskonflikt |
| Resultat och kassaflöde | Positiv EBIT och leverantörens FCF i samtliga fem årsrapporter |
| ROIC, leverantörens femårsgenomsnitt | Minst 25 % |
| ROIC, leverantörens femårslägsta | Minst 15 % |
| ROA-G, leverantörens femårsgenomsnitt | Minst 10 % |
| EBIT-marginal, median av fem årsrapporter | Minst 15 % |
| EBIT-marginal, lägsta av fem årsrapporter | Minst 10 % |
| Omsättning, årlig tillväxt mellan fem årsrapporter | Minst 5 % |
| Omsättning, årlig tillväxt mellan tre årsrapporter | Minst 3 % |
| Operativt kassaflöde, årlig tillväxt mellan fem årsrapporter | Minst 0 %; beräkningen kräver positiva värden i samtliga fem rapporter |
| Capex / operativt kassaflöde, leverantörens femårsgenomsnitt | 0–35 % |
| Leverantörens materiella tillgångar / senaste årsomsättning | 0–0,5 gånger |
| Nettoskuld / EBITDA, senaste leverantörsvärde | Högst 1,5 gånger; nettokassa tillåts |
| EBITDA-marginal, senaste leverantörsvärde | Strikt positiv |
| Starter Mid NPV / sparat pris | Minst 0 % |

Sorteringen prioriterar högsta **femårslägsta ROIC**, med bolagsnamn och ID som
skiljeregel. Saknade, ogiltiga eller ofullständiga nödvändiga värden passerar inte.
Gränserna bestämdes före genomgången av träffarnas namn och justerades inte för
att få med utvalda bolag.

Fem årsrapporter omfattar normalt ungefär **fyra års** förfluten tid; tre rapporter
omfattar ungefär två år. Appens tillväxttal använder verkliga rapportdatum och
jämförbar rapportvaluta. Leverantörens femårsfönster är en separat källa och
garanterar inte i sig att alla fem underliggande år är observerade.

## Tolkning och begränsningar

Detta identifierar bolag att undersöka, inte bevisat exceptionella företag.
Prissättningsmakt, kundernas beteende, konkurrensfördelar och framtida avkastning
på nyinvesteringar kräver bolagsanalys. ROIC är här [Börsdatas beräkning](https://borsdata.se/info/nyckeltal/roic),
inte normaliserad NOPAT dividerad med genomsnittligt materiellt rörelsekapital.
Små kapitalbaser, avskrivna tillgångar och kostnadsförd utveckling kan höja kvoten.
[Capex-måttet](https://borsdata.se/en/info/ratios/capex) omfattar även förvärv och
avyttringar och fastställer inte underhållsinvesteringar.

Kontrollera ersättningsinvesteringar, rörelsekapital, leasing, förvärv, utspädning,
kund- och regulatoriska risker samt likviditetsuthållighet. Ett krav på låg
tillgångsintensitet väljer också bort vissa exceptionella men kapitalintensiva
verksamheter. Omsättningstillväxten är nominell och kan innehålla förvärv.

## Daterad körning

Forskningsunderlaget är från 13 september 2026 och leverantörens KPI-underlag
från 10 augusti 2026. Det fasta kvalitetsurvalet ger 64 noteringar före
värderingsgränsen och tre efter: Moury Construct (Belgien och Tyskland) samt
Evolution (Sverige). De sparade kurserna är från 5–9 februari 2026.

Samtliga tre har negativ NPV med halverad terminaldel och i Low-scenariot.
Mourys Low-scenario ligger omkring −95 % och Evolutions omkring −32 % relativt
det sparade priset. Dessa modellkänsligheter måste granskas; de är inte bedömda
sannolikheter för förlust. Ingen aktuell undervärdering eller köprekommendation
följer av träffarna.

Oberoende Python-beräkning från hashkontrollerade artefakter stämmer med appens
parser, cellberäkningar, filter och sortering. Vyn sparades med appens Save view
och lästes tillbaka efter omladdning och full processomstart. Båda tidigare
sparade vyerna, bevakningslistorna och 63 andra localStorage-poster, inklusive
60 värderingsutkast, bevarades. Körningsunderlag och hashverifierad lokal
profilbackup finns i `desktop/test-results/exceptional-fair-screen-2026-09-28/`.
