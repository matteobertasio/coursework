import excel "C:\Users\matte\OneDrive - Università Commerciale Luigi Bocconi\Desktop\trade groupwork\ISIC_to_AG6_mapped_corrected.xlsx", sheet("Sheet1") firstrow

keep AG6_Code ISIC_Code
rename AG6_Code cmdCode
destring cmdCode, replace
tempfile mapping
save `mapping'

* 3. Torna al dataset principale e fai il merge
use "C:\Users\matte\OneDrive - Università Commerciale Luigi Bocconi\Desktop\trade groupwork\xbilat\BRU.dta", clear
destring cmdCode, replace
merge m:1 cmdCode using `mapping'

* 4. Controlla merge
drop if _merge == 2   // se qualche codice non matcha, lo elimini
drop _merge

* 5. Aggrega per: importatore, esportatore, settore
collapse (sum) primaryValue, by(reporterDesc partnerDesc ISIC_Code)

save "C:\Users\matte\OneDrive - Università Commerciale Luigi Bocconi\Desktop\trade groupwork\xbilat\BRU_bilateral_aggregated.dta", replace