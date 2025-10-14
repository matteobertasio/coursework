clear all
cd D:\econometrics\assignment
use assignment_data_group_17

*Question 1: 
sum n
sum w 
sum k
tab ys
sum ys

gen sector1w = w if sector == 1
gen sector2w = w if sector == 2
gen sector3w = w if sector == 3
gen sector4w = w if sector == 4

sum sector1w
sum sector2w
sum sector3w
sum sector4w


forvalues sector = 1/4 {
	gen sector_`sector'n = n if sector == `sector'
	sum sector_`sector'n
}

forvalues sector = 1/4 {
	gen sector_`sector'k = k if sector == `sector'
	sum sector_`sector'k
}

forvalues sector = 1/4 {
	gen sector_`sector'ys = ys if sector == `sector'
	sum sector_`sector'ys
}

forvalues sector = 1/4 {
	gen sector_`sector' = 1 if sector == `sector' 
	sum sector_`sector'
}

replace sector_1 = 0 if missing(sector_1)
replace sector_2 = 0 if missing(sector_2)
replace sector_3 = 0 if missing(sector_3)
replace sector_4 = 0 if missing(sector_4)

*Question 2:
reg n w k 
outreg2 using assighment1, tex label ctitle ("employment") replace

*Question 3:
test w
test k
test w k

*Question 4:
reg ys sector_2 sector_3 sector_4
reg w sector_2 sector_3 sector_4
reg k sector_2 sector_3 sector_4
reg n sector_2 sector_3 sector_4

*Test homogeneity
reg ys sector_2 sector_3 sector_4
test [sector_2] = [sector_3] = [sector_4]

reg w sector_2 sector_3 sector_4
test [sector_2] = [sector_3] = [sector_4]

reg k sector_2 sector_3 sector_4
test [sector_2] = [sector_3] = [sector_4]

reg n sector_2 sector_3 sector_4
test [sector_2] = [sector_3] = [sector_4]
*Prob>F= 0.0000 and so we can reject the null of homogeneity --> heterogeneity.

*Question 5
*accommadate heterogeneity by adding it as interaction term
reg n w k c.w#c.ys c.k#c.ys
outreg2 using assighment1, tex label ctitle ("employment") append

*group regression
bysort sector: reg n w k

reg n w k sector_2 sector_3 sector_4
outreg2 using assighment1, tex label ctitle ("employment") append

*Question 6
estat hettest
estat imtest, white
*Prob > chi2 = 0.0000 -> Heteroskedasticity

*Question 7
reg n w k,r


