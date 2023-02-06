IMPORT $;
Persons := $.File_Persons.File; 

r := RECORD
  Persons.Gender;
  CNT := COUNT(GROUP);
END;

XTAB_Persons_Gender := SORT(TABLE(Persons,r,Gender),Gender);

OUTPUT(XTAB_Persons_Gender);