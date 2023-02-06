IMPORT $;
Persons   := $.File_Persons.File;
Person_Layout := $.File_Persons.Layout;

NewPerson := RECORD
  UNSIGNED4 RecID;
	Person_Layout;
END;

NewPerson IDRecs(Persons Le,INTEGER Cnt) := TRANSFORM
 SELF.RecID := Cnt;
 SELF       := Le;
END;

EXPORT UID_Persons := PROJECT(Persons,IDRecs(LEFT,COUNTER)) 
                       :PERSIST('~CLASS::JDI::PERSIST::UID_People');