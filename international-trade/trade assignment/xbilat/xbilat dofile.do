levelsof sector, local(sectors)

tempname outfile
file open `outfile' using "xbilat1993.txt", write replace

foreach s of local sectors {
    preserve
    keep if sector == `s'
    
    sort id_dest id_orig
    
    * Loop su righe (paesi destinazione)
    forvalues i = 1/20 {
        * Loop su colonne (paesi origine)
        forvalues j = 1/20 {
            quietly summarize import if id_dest == `i' & id_orig == `j', meanonly
            local val = r(mean)
            if missing(`val') {
                local val = 0
            }
            file write `outfile' "`val' "
        }
        file write `outfile' _n
    }
    restore
}

file close `outfile'
