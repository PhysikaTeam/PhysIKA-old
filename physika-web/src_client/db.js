import Dexie from 'dexie'

const db = new Dexie('PhysikaDB');
db.version(1).stores({
    model:"++id,userID,[uploadDate+frameIndex]"
});

export default db;
