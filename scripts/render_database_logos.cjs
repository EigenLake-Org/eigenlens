// Compose official, locally stored SVG logos without changing their artwork.
// Optional documentation dependency: Node.js + sharp. See docs/assets/logos/README.md.
const fs=require('node:fs');
const path=require('node:path');
const sharp=require('sharp');
const assets=path.join(process.cwd(),'docs/assets');
(async()=>{
const width=1600,height=200;
const bg=Buffer.from(`<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}"><rect width="1600" height="200" rx="18" fill="#fff"/><g stroke="#e6ebef">${[320,640,960,1280].map(x=>`<path d="M${x} 48V152"/>`).join('')}</g><g font-family="Arial,sans-serif" fill="#233849"><text x="1391" y="97" font-size="23" font-weight="600">PostgreSQL</text><text x="1391" y="126" font-size="19" fill="#596d7d">+ pgvector</text></g></svg>`);
const overlays=[];
const specs=[['qdrant',0,242,72],['pinecone',1,250,72],['milvus',2,252,72],['weaviate',3,256,72]];
for(const [name,col,w,h] of specs){
 const {data,info}=await sharp(path.join(assets,'logos',name+'.svg')).resize({width:w,height:h,fit:'inside'}).png().toBuffer({resolveWithObject:true});
 overlays.push({input:data,left:Math.round(col*320+(320-info.width)/2),top:Math.round((height-info.height)/2)});
}
const pg=await sharp(path.join(assets,'logos/postgresql.svg')).resize({width:72,height:78,fit:'inside'}).png().toBuffer();
overlays.push({input:pg,left:1304,top:61});
await sharp(bg).composite(overlays).png().toFile(path.join(assets,'vector-databases.png'));
console.log('Rendered official logo strip.');
})();
