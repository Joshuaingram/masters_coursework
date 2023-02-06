IMPORT $;
Persons := $.File_Persons.File;

count_null := COUNT(Persons(DependentCount = 0));
count_total := COUNT(Persons);
pop_prop := (count_total-count_null)/count_total;
results := DATASET([{'Total Records', count_total},
                     {'Recs=0', count_null},
                     {'Population Pct', pop_prop * 100}],
                     {STRING valuetype, INTEGER val});

OUTPUT(results)